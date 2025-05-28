import numpy as np
from acoustic_simulator import AcousticSimulator
from time_reversal import TimeReversal

num_transducers = np.int32(64)

grid_center_x = 250
# grid_size = (1000, 1000)

transducer_z = np.asarray([300 for _ in range(num_transducers)], dtype=np.int32)
# transducer_x = np.asarray([500 + i * 10 for i in range(num_transducers)], dtype=np.int32)

start_x = grid_center_x - (num_transducers // 2) * 8 # Adjust spacing (e.g., 8 pixels)
transducer_x = np.array([start_x + i * 8 for i in range(num_transducers)], dtype=np.int32)

acoustic_sim_args = {
    "transducer_z": transducer_z,
    "transducer_x": transducer_x,
    "num_transducers": num_transducers,
}

time_reversal_args = {
    "transducer_z": transducer_z,
    "transducer_x": transducer_x,
    "num_transducers": num_transducers,
    "recordings_folder": "./AcousticSim",
}
sh = AcousticSimulator(**acoustic_sim_args)
# tr = TimeReversal(**time_reversal_args)