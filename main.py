import numpy as np
from acoustic_simulator import AcousticSimulator
from time_reversal import TimeReversal

num_transducers = np.int32(32)

transducer_z = np.asarray([800 for _ in range(num_transducers)], dtype=np.int32)
transducer_x = np.asarray([500 + i * 10 for i in range(num_transducers)], dtype=np.int32)

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
tr = TimeReversal(**time_reversal_args)