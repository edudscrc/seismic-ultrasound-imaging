import numpy as np
from webgpu_handler import WebGpuHandler
import matplotlib.pyplot as plt
from simulation_handler import SimulationHandler
from pathlib import Path
import re
import os


class TimeReversal(SimulationHandler):
    def __init__(self, **args):
        super().__init__()

        os.makedirs("./plots_tr", exist_ok=True)

        self.folder = Path("./TimeReversalSim")
        self.folder.mkdir(parents=True, exist_ok=True)

        acoustic_sim_folder = Path(args["recordings_folder"])

        self.bscan = np.load(f"{acoustic_sim_folder}/recordings.npy")

        self.transducer_z = args["transducer_z"]
        self.transducer_x = args["transducer_x"]
        self.num_transducers = args["num_transducers"]

        # source = np.load('./source.npy').astype(np.float32)
        # if len(source) < self.total_time:
        #     source = np.pad(source, (0, self.total_time - len(source)), 'constant').astype(np.float32)
        # elif len(source) > self.total_time:
        #     source = source[:self.total_time]

        # source_idx = ~np.isclose(source, 0)

        # # Cut the recorded source
        # self.bscan[:, ~np.isclose(source_idx, 0)] = np.float32(0)

        # Flip bscan
        self.flipped_bscan = self.bscan[:, ::-1]

        # WebGPU Buffer
        self.info_i32 = np.array(
            [
                self.grid_size_z,
                self.grid_size_x,
                self.num_transducers,
                0,
            ],
            dtype=np.int32
        )

        self.wgpu_handler = WebGpuHandler(shader_file='./time_reversal_sim.wgsl', wsz=self.grid_size_z, wsx=self.grid_size_x)
        
        # Inject flipped microphones code into shader string
        matches = re.findall(r'@binding\((\d+)\)', self.wgpu_handler.shader_string)
        last_binding = int(matches[-1])
        aux_string = ''
        for i in range(self.num_transducers):
            aux_string += f'''@group(0) @binding({i + (last_binding + 1)})
            var<storage,read> flipped_microphone_{i}: array<f32>;\n\n'''
        self.wgpu_handler.shader_string = self.wgpu_handler.shader_string.replace('//FLIPPED_MICROPHONES_BINDINGS', aux_string)
        aux_string = ''
        for i in range(self.num_transducers):
            aux_string += f'''if (microphone_index == {i})
                    {{
                        p_next[zx(z, x)] += flipped_microphone_{i}[infoI32.i];
                    }}\n'''
        self.wgpu_handler.shader_string = self.wgpu_handler.shader_string.replace('//FLIPPED_MICROPHONES_SIM', aux_string)
        self.wgpu_handler.create_shader_module()

        # Data passed to gpu buffers
        wgsl_data = {
            'infoI32': self.info_i32,
            'infoF32': self.info_f32,
            'c': self.c,
            'microphone_z': np.ascontiguousarray(self.transducer_z),
            'microphone_x': np.ascontiguousarray(self.transducer_x),
            'p_next': self.p_next,
            'p_current': self.p_current,
            'p_previous': self.p_previous,
            'dp_1_z': self.dp_1_z,
            'dp_1_x': self.dp_1_x,
            'dp_2_z': self.dp_2_z,
            'dp_2_x': self.dp_2_x,
            'psi_z': self.psi_z,
            'psi_x': self.psi_x,
            'phi_z': self.phi_z,
            'phi_x': self.phi_x,
            'absorption_z': self.absorption_z,
            'absorption_x': self.absorption_x,
            'is_z_absorption': self.is_z_absorption_int,
            'is_x_absorption': self.is_x_absorption_int,
            **{f"flipped_microphone_{i}": np.ascontiguousarray(self.flipped_bscan[i]) for i in range(self.num_transducers)}
        }

        self.wgpu_handler.create_buffers(wgsl_data)

        compute_forward_diff = self.wgpu_handler.create_compute_pipeline("forward_diff")
        compute_after_forward = self.wgpu_handler.create_compute_pipeline("after_forward")
        compute_backward_diff = self.wgpu_handler.create_compute_pipeline("backward_diff")
        compute_after_backward = self.wgpu_handler.create_compute_pipeline("after_backward")
        compute_sim = self.wgpu_handler.create_compute_pipeline("sim")
        compute_incr_time = self.wgpu_handler.create_compute_pipeline("incr_time")

        l2_norm = np.zeros(self.grid_size_shape, dtype=np.float32)

        for i in range(self.total_time):
            command_encoder = self.wgpu_handler.device.create_command_encoder()
            compute_pass = command_encoder.begin_compute_pass()

            for index, bind_group in enumerate(self.wgpu_handler.bind_groups):
                compute_pass.set_bind_group(index, bind_group, [], 0, 999999)

            compute_pass.set_pipeline(compute_forward_diff)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_after_forward)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_backward_diff)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_after_backward)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_sim)
            compute_pass.dispatch_workgroups(self.grid_size_z // self.wgpu_handler.ws[0],
                                             self.grid_size_x // self.wgpu_handler.ws[1])

            compute_pass.set_pipeline(compute_incr_time)
            compute_pass.dispatch_workgroups(1)

            compute_pass.end()
            self.wgpu_handler.device.queue.submit([command_encoder.finish()])

            """ READ BUFFERS """
            self.p_next = (np.asarray(self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b5']).cast("f"))
                             .reshape(self.grid_size_shape))

            l2_norm += np.square(self.p_next)
            
            # if i % 50 == 0:
            #     plt.imsave(f'./plots_tr/pf_{i}.png', self.p_next, cmap='bwr')

        plt.imsave(f"{self.folder}/l2_norm.png", np.sqrt(l2_norm), vmax=np.amax(np.sqrt(l2_norm)) / 2)
        np.save(f"{self.folder}/l2_norm.npy", np.sqrt(l2_norm))
