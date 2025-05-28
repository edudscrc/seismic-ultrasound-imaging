import numpy as np
from webgpu_handler import WebGpuHandler
import matplotlib.pyplot as plt
from simulation_handler import SimulationHandler
import os
from pathlib import Path
from time import perf_counter


class AcousticSimulator(SimulationHandler):
    def __init__(self, **args):
        super().__init__()

        os.makedirs("./plots", exist_ok=True)

        self.folder = Path("./AcousticSim")
        self.folder.mkdir(parents=True, exist_ok=True)

        self.transducer_z = args["transducer_z"]
        self.transducer_x = args["transducer_x"]
        self.num_transducers = args["num_transducers"]

        self.recordings = np.asarray([[0 for _ in range(self.total_time)] for _ in range(self.num_transducers)], dtype=np.float32)

        self.source_z = 400
        self.source_x = 800

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
            ],
            dtype=np.int32
        )

        self.wgpu_handler = WebGpuHandler()

        self.wgpu_handler.create_shader_module("./synthetic_acou_sim.wgsl", self.grid_size_shape, (8, 8))

        nbytes = int(self.grid_size_x * self.grid_size_z * 4)

        # Data passed to gpu buffers
        wgsl_data = {
            'p_next': (nbytes, True),
            'p_current': (nbytes, True),
            'p_previous': (nbytes, True),
            'dp_1_z': (nbytes, True),
            'dp_1_x': (nbytes, True),
            'dp_2_z': (nbytes, True),
            'dp_2_x': (nbytes, True),
            'phi_z': (nbytes, True),
            'phi_x': (nbytes, True),
            'psi_z': (nbytes, True),
            'psi_x': (nbytes, True),
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

        time_st = perf_counter()
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

            """ READ BUFFERS """
            self.p_next = np.asarray(self.wgpu_handler.device.queue.read_buffer(self.wgpu_handler.buffers[0]).cast("f")).reshape(self.grid_size_shape)

            self.recordings[:, i] = self.p_next[self.transducer_z[:], self.transducer_x[:]]

            if i % 50 == 0:
                plt.imsave(f'./plots/pf_{i}.png', self.p_next, cmap='bwr')

        np.save(f"{self.folder}/recordings.npy", self.recordings)

        time_end = perf_counter()

        print(f'Total time: {time_end - time_st} seconds')

        print('Synthetic Acoustic Simulation finished.')
