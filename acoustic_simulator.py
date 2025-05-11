import numpy as np
from webgpu_handler import WebGpuHandler
import matplotlib.pyplot as plt
from simulation_handler import SimulationHandler
import os
from pathlib import Path


class AcousticSimulator(SimulationHandler):
    def __init__(self, **args):
        super().__init__()

        self.folder = Path("./AcousticSim")
        self.folder.mkdir(parents=True, exist_ok=True)

        self.transducer_z = args["transducer_z"]
        self.transducer_x = args["transducer_x"]
        self.num_transducers = args["num_transducers"]

        self.recordings = np.asarray([[0 for _ in range(self.total_time)] for _ in range(self.num_transducers)], dtype=np.float32)

        self.source_z = 500
        self.source_x = 500

        self.source = np.load('./source.npy').astype(np.float32)
        if len(self.source) < self.total_time:
            self.source = np.pad(self.source, (0, self.total_time - len(self.source)), 'constant').astype(np.float32)
        elif len(self.source) > self.total_time:
            self.source = self.source[:self.total_time]
        
        self.info_i32 = np.array(
            [
                self.grid_size_z,
                self.grid_size_x,
                self.source_z,
                self.source_x,
                0,
            ],
            dtype=np.int32
        )

        self.wgpu_handler = WebGpuHandler(shader_file='./synthetic_acou_sim.wgsl', wsz=self.grid_size_z, wsx=self.grid_size_x)

        self.wgpu_handler.create_shader_module()

        # Data passed to gpu buffers
        wgsl_data = {
            'infoI32': self.info_i32,
            'infoF32': self.info_f32,
            'c': self.c,
            'source': self.source,
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
        }

        self.wgpu_handler.create_buffers(wgsl_data)

        compute_forward_diff = self.wgpu_handler.create_compute_pipeline("forward_diff")
        compute_after_forward = self.wgpu_handler.create_compute_pipeline("after_forward")
        compute_backward_diff = self.wgpu_handler.create_compute_pipeline("backward_diff")
        compute_after_backward = self.wgpu_handler.create_compute_pipeline("after_backward")
        compute_sim = self.wgpu_handler.create_compute_pipeline("sim")
        compute_incr_time = self.wgpu_handler.create_compute_pipeline("incr_time")

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
            self.p_next = (np.asarray(self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers['b4']).cast("f"))
                             .reshape(self.grid_size_shape))

            self.recordings[:, i] = self.p_next[self.transducer_z[:], self.transducer_x[:]]

            if i % 50 == 0:
                plt.imsave(f'./plots/pf_{i}.png', self.p_next, cmap='bwr')

        np.save(f"{self.folder}/recordings.npy", self.recordings)

        print('Synthetic Acoustic Simulation finished.')