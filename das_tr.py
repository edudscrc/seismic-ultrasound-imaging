import numpy as np
from webgpu_handler import WebGpuHandler
import matplotlib.pyplot as plt
from das_simulation_handler import DAS_SimulationHandler
from pathlib import Path
import re


class DAS_TimeReversal(DAS_SimulationHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.plots_folder = Path("./plots_tr")
        self.plots_folder.mkdir(parents=True, exist_ok=True)

        for item in self.plots_folder.iterdir():
            item.unlink()

        self.folder = Path("./TimeReversalSim")
        self.folder.mkdir(parents=True, exist_ok=True)

        for item in self.folder.iterdir():
            item.unlink()

        self.bscan = kwargs['bscan']

        self.bscan[:, :400] = np.float32(0)

        # Flip bscan
        self.flipped_bscan = self.bscan[:, ::-1]
        # Normalize bscan
        self.flipped_bscan = self.flipped_bscan / np.amax(np.abs(self.flipped_bscan))

        increment_time = 9000
        self.total_time += increment_time
        self.flipped_bscan = np.concatenate((self.flipped_bscan, np.zeros((self.flipped_bscan.shape[0], increment_time), dtype=np.float32)), axis=1)

        self.cmap_vmax = 3
        self.cmap_vmin = -self.cmap_vmax

        # WebGPU Buffer
        self.info_i32 = np.array(
            [
                self.grid_size_z,
                self.grid_size_x,
                self.num_transducers,
            ],
            dtype=np.int32
        )

        shader_string = Path("./time_reversal_sim.wgsl").read_text()
        
        # Inject flipped microphones code into shader string
        matches = re.findall(r'@binding\((\d+)\)', shader_string)
        aux_string = ''
        for i in range(self.num_transducers):
            aux_string += f'''@group(2) @binding({i})
var<storage,read> flipped_recording_{i}: array<f32>;\n\n'''
        shader_string = shader_string.replace('//FLIPPED_MICROPHONES_BINDINGS', aux_string)
        aux_string = ''
        for i in range(self.num_transducers):
            aux_string += f'''if (transducer_index == {i})
            {{
                p_next[zx(z, x)] += flipped_recording_{i}[i];
            }}\n'''
        shader_string = shader_string.replace('//FLIPPED_MICROPHONES_SIM', aux_string)

        with open("injected_tr.wgsl", 'w', encoding='utf-8') as file:
            file.write(shader_string)

        self.wgpu_handler = WebGpuHandler()
        self.wgpu_handler.create_shader_module("injected_tr.wgsl", self.grid_size_shape, (8, 8))

        # Data passed to gpu buffers
        wgsl_data = {
            'p_next': (self.roi_nbytes, True),
            'p_current': (self.roi_nbytes, True),
            'p_previous': (self.roi_nbytes, True),
            'dp_1_z': (self.roi_nbytes, True),
            'dp_1_x': (self.roi_nbytes, True),
            'dp_2_z': (self.roi_nbytes, True),
            'dp_2_x': (self.roi_nbytes, True),
            'phi_z': (self.roi_nbytes, True),
            'phi_x': (self.roi_nbytes, True),
            'psi_z': (self.roi_nbytes, True),
            'psi_x': (self.roi_nbytes, True),
            'infoI32': (self.info_i32, False),
            'infoF32': (self.info_f32, False),
            'c': (self.c, False),
            'absorption_z': (self.absorption_z, False),
            'absorption_x': (self.absorption_x, False),
            'is_z_absorption': (self.is_z_absorption_int, False),
            'is_x_absorption': (self.is_x_absorption_int, False),
            'transducer_z': (np.ascontiguousarray(self.transducer_z), False),
            'transducer_x': (np.ascontiguousarray(self.transducer_x), False),
            'i': (np.int32(0), False),
            **{f"flipped_recording_{i}": (np.ascontiguousarray(self.flipped_bscan[i]), False) for i in range(self.num_transducers)},
        }

        self.wgpu_handler.set_buffers(wgsl_data, "p_next")
        self.wgpu_handler.create_buffers(debug=False)
        self.wgpu_handler.create_bind_group_layouts()
        self.wgpu_handler.create_pipeline_layout()
        self.wgpu_handler.create_bind_groups()

        forward_diff = self.wgpu_handler.create_compute_pipeline("forward_diff")
        apply_cpml_to_first_order_diff = self.wgpu_handler.create_compute_pipeline("apply_cpml_to_first_order_diff")
        backward_diff = self.wgpu_handler.create_compute_pipeline("backward_diff")
        apply_cpml_to_second_order_diff = self.wgpu_handler.create_compute_pipeline("apply_cpml_to_second_order_diff")
        simulate = self.wgpu_handler.create_compute_pipeline("simulate")
        increment_time = self.wgpu_handler.create_compute_pipeline("increment_time")

        l2_norm = np.zeros(self.grid_size_shape, dtype=np.float32)

        for i in range(self.total_time):
            command_encoder = self.wgpu_handler.device.create_command_encoder()
            compute_pass = command_encoder.begin_compute_pass()

            for index, bind_group in enumerate(self.wgpu_handler.bind_groups):
                compute_pass.set_bind_group(index, bind_group, [])

            self.wgpu_handler.dispatch_workgroups_to_pipeline(compute_pass, forward_diff)
            self.wgpu_handler.dispatch_workgroups_to_pipeline(compute_pass, apply_cpml_to_first_order_diff)
            self.wgpu_handler.dispatch_workgroups_to_pipeline(compute_pass, backward_diff)
            self.wgpu_handler.dispatch_workgroups_to_pipeline(compute_pass, apply_cpml_to_second_order_diff)
            self.wgpu_handler.dispatch_workgroups_to_pipeline(compute_pass, simulate)
            self.wgpu_handler.dispatch_workgroups_to_pipeline(compute_pass, increment_time, [1])

            compute_pass.end()
            self.wgpu_handler.device.queue.submit([command_encoder.finish()])

            self.p_next = self.wgpu_handler.read_buffer(group=0, binding=0)
            self.p_next = np.frombuffer(self.p_next, dtype=np.float32).reshape(self.grid_size_shape)

            l2_norm += np.square(self.p_next)
            
            if i == 0 or (i + 1) % 50 == 0:
                print(f"Simulated {i + 1}/{self.total_time}")
                plt.figure()
                plt.scatter(self.transducer_x, self.transducer_z, s=0.1)
                plt.imshow(self.p_next, cmap='coolwarm', aspect='auto', vmax=self.cmap_vmax, vmin=self.cmap_vmin)
                plt.colorbar()
                plt.savefig(f'{self.plots_folder}/pf_{i}.png', dpi=300)
                plt.close()

        l2_norm = np.sqrt(l2_norm)

        np.save("l2_norm.npy", l2_norm)

        plt.figure()
        plt.imshow(l2_norm, aspect='auto')
        plt.colorbar()
        plt.title("L2-Norm - Time Reversal")
        plt.show()

        print('Time Reversal Simulation finished.')
