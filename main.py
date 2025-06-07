import numpy as np
from acoustic_simulator import AcousticSimulator
from time_reversal import TimeReversal

num_transducers = np.int32(64)

grid_center_x = 500

transducer_z = np.asarray([500 for _ in range(num_transducers)], dtype=np.int32)

start_x = grid_center_x - (num_transducers // 2) * 8
transducer_x = np.array([start_x + i * 8 for i in range(num_transducers)], dtype=np.int32)

medium_c = np.float32(1500)
c = np.full(shape=(1000, 1000), fill_value=medium_c, dtype=np.float32)
c[800, 300:700] = np.float32(0)

global_sim_params = {
    'total_time': 3000,
    'grid_size_z': 1000,
    'grid_size_x': 1000,
    'num_transducers': 64,
    'transducer_z': transducer_z,
    'transducer_x': transducer_x,
    'dt': 5e-7,
    'dz': 1.5e-3,
    'dx': 1.5e-3,
    'cpml_absorption_layer_size': 25,
    'damping_coefficient': 3e6,
    "transducer_z": transducer_z,
    "transducer_x": transducer_x,
    "num_transducers": num_transducers,
}

acoustic_sim_params = {
    'source_z': 300,
    'source_x': 500,
    'c': c,
}

time_reversal_params = {
    "recordings_folder": "./AcousticSim",
    'c': np.full(shape=(1000, 1000), fill_value=medium_c, dtype=np.float32),
}

acoustic_sim_params.update(global_sim_params)
time_reversal_params.update(global_sim_params)

sh = AcousticSimulator(**acoustic_sim_params)
tr = TimeReversal(**time_reversal_params)
