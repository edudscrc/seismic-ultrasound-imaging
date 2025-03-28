import numpy as np
from webgpu_handler import WebGpuHandler
import matplotlib.pyplot as plt


class SimulationHandler:
    def __init__(self):

        # dt = 1 / fs
        # fs = 1 / dt
        # (s/px)
        self.dt = np.float32(5e-7)

        # Spatial sampling (m/px)
        self.dz = np.float32(1.5e-3)
        self.dx = np.float32(1.5e-3)

        # Size of each axis in pixels
        self.grid_size_z = np.int32(1000)
        self.grid_size_x = np.int32(1000)
        self.grid_size_shape = (self.grid_size_z, self.grid_size_x)

        # Speed (m/s)
        self.c = np.full(shape=self.grid_size_shape, fill_value=1500, dtype=np.float32)

        # Total time (total amount of frames)
        self.total_time = np.int32(1000)

        # Courant
        self.CFL = np.amax(self.c) * self.dt * ((1 / self.dz) + (1 / self.dx))

        # Pressure fields
        self.p_next = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.p_current = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.p_previous = np.zeros(self.grid_size_shape, dtype=np.float32)

        # Partial derivatives
        self.dp_1_z = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_1_x = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_2_z = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.dp_2_x = np.zeros(self.grid_size_shape, dtype=np.float32)

        """ CPML """
        # self.absorption_layer_size = np.int32(25)
        self.absorption_layer_size = np.int32(25)
        self.damping_coefficient = np.float32(3e6)

        x, z = np.meshgrid(np.arange(self.grid_size_x, dtype=np.float32), np.arange(self.grid_size_z, dtype=np.float32))

        # Choose absorbing boundaries
        # self.is_z_absorption = np.array([False for _ in range(int(self.grid_size_z * self.grid_size_x))]).reshape(self.grid_size_shape)
        self.is_z_absorption = (z > self.grid_size_z - self.absorption_layer_size)
        self.is_x_absorption = (x > self.grid_size_x - self.absorption_layer_size) | (x < self.absorption_layer_size)

        self.absorption_coefficient = np.exp(
            -(self.damping_coefficient * (np.arange(self.absorption_layer_size) / self.absorption_layer_size) ** 2) * self.dt
        ).astype(np.float32)

        self.psi_z = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.psi_x = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.phi_z = np.zeros(self.grid_size_shape, dtype=np.float32)
        self.phi_x = np.zeros(self.grid_size_shape, dtype=np.float32)

        self.absorption_z = np.ones(self.grid_size_shape, dtype=np.float32)
        self.absorption_x = np.ones(self.grid_size_shape, dtype=np.float32)

        # self.absorption_z[:self.absorption_layer_size, :] = self.absorption_coefficient[:, np.newaxis][::-1]  # z < layer_size
        self.absorption_z[-self.absorption_layer_size:, :] = self.absorption_coefficient[:, np.newaxis]  # z > (size_z - layer_size)
        self.absorption_x[:, :self.absorption_layer_size] = self.absorption_coefficient[::-1]  # x < layer_size
        self.absorption_x[:, -self.absorption_layer_size:] = self.absorption_coefficient  # x > (size_x - layer_size)

        # Converts boolean array to int array to pass to GPU
        self.is_z_absorption_int = self.is_z_absorption.astype(np.int32)
        self.is_x_absorption_int = self.is_x_absorption.astype(np.int32)

        # self.transducer_z = None
        # self.transducer_x = None
        # self.transducers_amount = None

        self.source_z = 500
        self.source_x = 500

        # self.reflector_z, self.reflector_x = np.where(self.c == 0)
        # self.reflectors_amount = None

        # self.transducers_recording = np.array([[0 for _ in range(self.total_time)] for _ in range(self.transducers_amount)], dtype=np.float32)

        # Source
        self.source = np.load('./source.npy').astype(np.float32)
        if len(self.source) < self.total_time:
            self.source = np.pad(self.source, (0, self.total_time - len(self.source)), 'constant').astype(np.float32)
        elif len(self.source) > self.total_time:
            self.source = self.source[:self.total_time]

        # WebGPU buffer
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

        # WebGPU buffer
        self.info_f32 = np.array(
            [
                self.dz,
                self.dx,
                self.dt,
            ],
            dtype=np.float32
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

            # self.microphones_recording[:, i] = self.p_next[self.microphone_z[:], self.microphone_x[:]]

            if i % 50 == 0:
                plt.imsave(f'./plots/pf_{i}.png', self.p_next, cmap='bwr')

        print('Synthetic Acoustic Simulation finished.')
    