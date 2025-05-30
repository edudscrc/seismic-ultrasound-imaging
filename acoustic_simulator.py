import numpy as np
from webgpu_handler import WebGpuHandler
import matplotlib.pyplot as plt
from simulation_handler import SimulationHandler
import os
from pathlib import Path
from scipy.signal.windows import gaussian


class AcousticSimulator(SimulationHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        os.makedirs("./plots", exist_ok=True)

        self.folder = Path("./AcousticSim")
        self.folder.mkdir(parents=True, exist_ok=True)

        self.recordings = np.asarray([[0 for _ in range(self.total_time)] for _ in range(self.num_transducers)], dtype=np.float32)

        self.source_z = kwargs["source_z"]
        self.source_x = kwargs["source_x"]

        self.source = gaussian(self.total_time, 1.5)
        self.source = np.roll(self.source, -1 * int(self.total_time / 2 - 20)).astype(np.float32)

        self.info_i32 = np.array(
            [
                self.grid_size_z,
                self.grid_size_x,
                self.source_z,
                self.source_x,
            ],
            dtype=np.int32
        )

        self.wgpu_handler = WebGpuHandler()
        self.wgpu_handler.create_shader_module("./synthetic_acou_sim.wgsl", self.grid_size_shape, (8, 8))

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
            'source': (self.source, False),
            'absorption_z': (self.absorption_z, False),
            'absorption_x': (self.absorption_x, False),
            'is_z_absorption': (self.is_z_absorption_int, False),
            'is_x_absorption': (self.is_x_absorption_int, False),
            'i': (np.int32(0), False),
        }

        self.wgpu_handler.set_buffers(wgsl_data, "p_next")
        self.wgpu_handler.create_buffers()
        self.wgpu_handler.create_bind_group_layouts()
        self.wgpu_handler.create_pipeline_layout()
        self.wgpu_handler.create_bind_groups()

        forward_diff = self.wgpu_handler.create_compute_pipeline("forward_diff")
        apply_cpml_to_first_order_diff = self.wgpu_handler.create_compute_pipeline("apply_cpml_to_first_order_diff")
        backward_diff = self.wgpu_handler.create_compute_pipeline("backward_diff")
        apply_cpml_to_second_order_diff = self.wgpu_handler.create_compute_pipeline("apply_cpml_to_second_order_diff")
        simulate = self.wgpu_handler.create_compute_pipeline("simulate")
        increment_time = self.wgpu_handler.create_compute_pipeline("increment_time")

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

            self.recordings[:, i] = self.p_next[self.transducer_z[:], self.transducer_x[:]]

            if i % 50 == 0:
                plt.figure()
                plt.scatter(self.transducer_x, self.transducer_z, s=0.05)
                plt.scatter(self.source_x, self.source_z, s=0.05)
                plt.imshow(self.p_next, cmap='bwr')
                plt.savefig(f'./plots/pf_{i}.png', dpi=300)
                plt.close()

        np.save(f"{self.folder}/recordings.npy", self.recordings)

        print('Acoustic Simulation finished.')
