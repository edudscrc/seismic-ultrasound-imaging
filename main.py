import numpy as np
from acoustic_simulator import AcousticSimulator
from time_reversal import TimeReversal

num_transducers = np.int32(64)

grid_size = (1500, 1500)

grid_center_x = int(grid_size[0] // 2)

transducer_z = np.asarray([500 for _ in range(num_transducers)], dtype=np.int32)

start_x = grid_center_x - (num_transducers // 2) * 8
transducer_x = np.array([start_x + i * 8 for i in range(num_transducers)], dtype=np.int32)

medium_c = np.float32(1500)

c = np.full(shape=(grid_size[0], grid_size[1]), fill_value=medium_c, dtype=np.float32)

c_with_reflectors = c.copy()
c_with_reflectors[1300, 1100] = np.float32(0)

dz = 1.5e-3
dx = 1.5e-3

N = 2
cpml_absorption_layer_size = 50
R_c = 0.001
d0 = - ( (N+1) * medium_c ) / [ 2 * (cpml_absorption_layer_size * dx) ] * np.log(R_c)

global_sim_params = {
    'total_time': 5000,
    'grid_size_z': grid_size[0],
    'grid_size_x': grid_size[1],
    'num_transducers': num_transducers,
    'transducer_z': transducer_z,
    'transducer_x': transducer_x,
    'dt': 5e-7,
    'dz': dz,
    'dx': dx,
    'cpml_absorption_layer_size': cpml_absorption_layer_size,
    'damping_coefficient': d0,
    "c_with_reflectors": c_with_reflectors,
    "c": c
}

# Modes:
# 0 -> Acoustic Simulation
# 1 -> Time Reversal

acoustic_sim_params = {
    'mode': 0,
    'source_z': 300,
    'source_x': 500,
}

time_reversal_params = {
    'mode': 1,
    "recordings_folder": "./AcousticSim",
}

acoustic_sim_params.update(global_sim_params)
time_reversal_params.update(global_sim_params)

sh = AcousticSimulator(**acoustic_sim_params)
tr = TimeReversal(**time_reversal_params)
