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
        self.absorption_layer_size = np.int32(25)
        self.damping_coefficient = np.float32(3e6)
        x, z = np.meshgrid(np.arange(self.grid_size_x, dtype=np.float32), np.arange(self.grid_size_z, dtype=np.float32))

        # Choose absorbing boundaries
        # self.is_z_absorption = np.array([False for _ in range(int(self.grid_size_z * self.grid_size_x))]).reshape(self.grid_size_shape)
        self.is_z_absorption = (z > self.grid_size_z - self.absorption_layer_size) | (z < self.absorption_layer_size)
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

        self.absorption_z[:self.absorption_layer_size, :] = self.absorption_coefficient[:, np.newaxis][::-1]  # z < layer_size
        self.absorption_z[-self.absorption_layer_size:, :] = self.absorption_coefficient[:, np.newaxis]  # z > (size_z - layer_size)
        self.absorption_x[:, :self.absorption_layer_size] = self.absorption_coefficient[::-1]  # x < layer_size
        self.absorption_x[:, -self.absorption_layer_size:] = self.absorption_coefficient  # x > (size_x - layer_size)

        # Converts boolean array to int array to pass to GPU
        self.is_z_absorption_int = self.is_z_absorption.astype(np.int32)
        self.is_x_absorption_int = self.is_x_absorption.astype(np.int32)

        # self.transducer_z = None
        # self.transducer_x = None
        # self.transducers_amount = None

        # self.reflector_z, self.reflector_x = np.where(self.c == 0)
        # self.reflectors_amount = None

        # self.transducers_recording = np.array([[0 for _ in range(self.total_time)] for _ in range(self.transducers_amount)], dtype=np.float32)

        # WebGPU buffer
        self.info_f32 = np.array(
            [
                self.dz,
                self.dx,
                self.dt,
            ],
            dtype=np.float32
        )
