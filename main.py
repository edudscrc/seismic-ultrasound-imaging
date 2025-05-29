import numpy as np
from acoustic_simulator import AcousticSimulator
from time_reversal import TimeReversal
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian

num_transducers = np.int32(64)

grid_center_x = 500
# grid_size = (1000, 1000)

transducer_z = np.asarray([300 for _ in range(num_transducers)], dtype=np.int32)
# transducer_x = np.asarray([500 + i * 10 for i in range(num_transducers)], dtype=np.int32)

start_x = grid_center_x - (num_transducers // 2) * 8 # Adjust spacing (e.g., 8 pixels)
transducer_x = np.array([start_x + i * 8 for i in range(num_transducers)], dtype=np.int32)

global_sim_params = {
    'total_time': 1000,
    'grid_size_z': 1000,
    'grid_size_x': 1000,
    'num_transducers': 64,
    'transducer_z': transducer_z,
    'transducer_x': transducer_x,
    'dt': 5e-7,
    'dz': 1.5e-3,
    'dx': 1.5e-3,
    'c': 1500,
    'cpml_absorption_layer_size': 25,
    'damping_coefficient': 3e6,
    "transducer_z": transducer_z,
    "transducer_x": transducer_x,
    "num_transducers": num_transducers,
}

acoustic_sim_params = {
    'source_z': 500,
    'source_x': 500,
}

acoustic_sim_params.update(global_sim_params)

time_reversal_args = {
    "recordings_folder": "./AcousticSim",
}

sh = AcousticSimulator(**acoustic_sim_params)
# tr = TimeReversal(**time_reversal_args)
