import numpy as np
from das_tr import DAS_TimeReversal

bscan = np.load('./aquisicao_40km_50ns_21_10_2024_ds19_17500m_24000m_40705s_40715s_fs900Hz.npy')

spatial_start = 17500
spatial_end = 24000

acquisition_step = np.float32(1. / 400e6)
nu = 3e8 / 1.4682
dx_original = np.float32((acquisition_step * (nu / 2.)))

offset_idx = 300
apex_idx = 696

fs = 900
dx = np.float32((spatial_end - spatial_start) / bscan.shape[0])
dz = dx
dt = np.float32(1 / fs)

print(f'{dx = }')

num_transducers = int(offset_idx * 2)
total_time = bscan.shape[1]

# Grid em metros
size_meters_z = np.float32((spatial_end - spatial_start) + 10000)
size_meters_x = np.float32((spatial_end - spatial_start) + 10000)

grid_size_z = np.int32(size_meters_z / dz)
grid_size_x = np.int32(size_meters_x / dx)
grid_size_shape = (grid_size_z, grid_size_x)

print(f'{grid_size_shape = }')

# Microphones' position
transducer_x = []
for rp in range(num_transducers):
    transducer_x.append((dx * rp) / dx)
transducer_x = (np.int32(np.asarray(transducer_x))
                    + np.int32((grid_size_x - transducer_x[-1]) / 2))

transducer_z = np.full(num_transducers, 100, dtype=np.int32)  # Não colocar microfones no índice 0.

# Speed (m/s)
c_water = np.float32(1500)
c = np.full(grid_size_shape, fill_value=c_water, dtype=np.float32)

N = 2
cpml_absorption_layer_size = 50
R_c = 0.001
d0 = - ( (N+1) * c_water ) / [ 2 * (cpml_absorption_layer_size * dx) ] * np.log(R_c)

global_sim_params = {
    'dt': dt,
    'c': c,
    'dz': dz,
    'dx': dx,
    'grid_size_z': grid_size_z,
    'grid_size_x': grid_size_x,
    'total_time': total_time,

    'cpml_absorption_layer_size': cpml_absorption_layer_size,
    'damping_coefficient': d0,

    'num_transducers': num_transducers,

    'transducer_z': transducer_z,
    'transducer_x': transducer_x,
}

bscan = bscan[apex_idx-offset_idx:apex_idx+offset_idx, :]
print(bscan.shape)

tr_config = {
    'bscan': bscan
}

global_sim_params.update(tr_config)

tr_sim = DAS_TimeReversal(**global_sim_params)
