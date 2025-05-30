import numpy as np
from webgpu_handler import WebGpuHandler
import matplotlib.pyplot as plt


class SimulationHandler:
    def __init__(self, **kwargs):
        # (s/px)
        self.dt = np.float32(kwargs["dt"])

        # Spatial sampling (m/px)
        self.dz = np.float32(kwargs["dz"])
        self.dx = np.float32(kwargs["dx"])

        # Size of each axis in pixels
        self.grid_size_z = np.int32(kwargs["grid_size_z"])
        self.grid_size_x = np.int32(kwargs["grid_size_x"])
        self.grid_size_shape = (self.grid_size_z, self.grid_size_x)

        self.roi_nbytes = int(self.grid_size_z * self.grid_size_x * np.dtype(np.int32).itemsize)

        # Speed (m/s)
        self.c = np.full(shape=self.grid_size_shape, fill_value=kwargs["c"], dtype=np.float32)

        # Total time (total amount of frames)
        self.total_time = np.int32(kwargs["total_time"])

        self.transducer_z = kwargs["transducer_z"]
        self.transducer_x = kwargs["transducer_x"]
        self.num_transducers = kwargs["num_transducers"]

        # Courant
        self.CFL = np.amax(self.c) * self.dt * ((1 / self.dz) + (1 / self.dx))

        # Pressure field
        self.p_next = np.zeros(self.grid_size_shape, dtype=np.float32)

        """ CPML """
        self.absorption_layer_size = np.int32(kwargs["cpml_absorption_layer_size"])
        self.damping_coefficient = np.float32(kwargs["damping_coefficient"])
        x, z = np.meshgrid(np.arange(self.grid_size_x, dtype=np.float32), np.arange(self.grid_size_z, dtype=np.float32))

        # Choose absorbing boundaries
        self.is_z_absorption = (z > self.grid_size_z - self.absorption_layer_size) | (z < self.absorption_layer_size)
        self.is_x_absorption = (x > self.grid_size_x - self.absorption_layer_size) | (x < self.absorption_layer_size)

        self.absorption_coefficient = np.exp(
            -(self.damping_coefficient * (np.arange(self.absorption_layer_size) / self.absorption_layer_size) ** 2) * self.dt
        ).astype(np.float32)

        self.absorption_z = np.ones(self.grid_size_shape, dtype=np.float32)
        self.absorption_x = np.ones(self.grid_size_shape, dtype=np.float32)

        self.absorption_z[:self.absorption_layer_size, :] = self.absorption_coefficient[:, np.newaxis][::-1]  # z < layer_size
        self.absorption_z[-self.absorption_layer_size:, :] = self.absorption_coefficient[:, np.newaxis]  # z > (size_z - layer_size)
        self.absorption_x[:, :self.absorption_layer_size] = self.absorption_coefficient[::-1]  # x < layer_size
        self.absorption_x[:, -self.absorption_layer_size:] = self.absorption_coefficient  # x > (size_x - layer_size)

        # Converts boolean array to int array to pass to GPU
        self.is_z_absorption_int = self.is_z_absorption.astype(np.int32)
        self.is_x_absorption_int = self.is_x_absorption.astype(np.int32)

        # self.reflector_z, self.reflector_x = np.where(self.c == 0)
        # self.reflectors_amount = None

        # WebGPU buffer
        self.info_f32 = np.array(
            [
                self.dz,
                self.dx,
                self.dt,
            ],
            dtype=np.float32
        )
