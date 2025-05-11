struct InfoInt {
    grid_size_z: i32,
    grid_size_x: i32,
    microphones_amount: i32,
    i: i32,
};

struct InfoFloat {
    dz: f32,
    dx: f32,
    dt: f32,
};

@group(0) @binding(0)
var<storage,read_write> infoI32: InfoInt;

@group(0) @binding(1)
var<storage,read> infoF32: InfoFloat;

@group(0) @binding(2)
var<storage,read> c: array<f32>;

@group(0) @binding(3)
var<storage,read> microphone_z: array<i32>;

@group(0) @binding(4)
var<storage,read> microphone_x: array<i32>;

@group(0) @binding(5)
var<storage,read_write> p_next: array<f32>;

@group(0) @binding(6)
var<storage,read_write> p_current: array<f32>;

@group(0) @binding(7)
var<storage,read_write> p_previous: array<f32>;

@group(0) @binding(8)
var<storage,read_write> dp_1_z: array<f32>;

@group(0) @binding(9)
var<storage,read_write> dp_1_x: array<f32>;

@group(0) @binding(10)
var<storage,read_write> dp_2_z: array<f32>;

@group(0) @binding(11)
var<storage,read_write> dp_2_x: array<f32>;

@group(0) @binding(12)
var<storage,read_write> psi_z: array<f32>;

@group(0) @binding(13)
var<storage,read_write> psi_x: array<f32>;

@group(0) @binding(14)
var<storage,read_write> phi_z: array<f32>;

@group(0) @binding(15)
var<storage,read_write> phi_x: array<f32>;

@group(0) @binding(16)
var<storage,read> absorption_z: array<f32>;

@group(0) @binding(17)
var<storage,read> absorption_x: array<f32>;

@group(0) @binding(18)
var<storage,read> is_z_absorption: array<i32>;

@group(0) @binding(19)
var<storage,read> is_x_absorption: array<i32>;

//FLIPPED_MICROPHONES_BINDINGS

// 2D index to 1D index
fn zx(z: i32, x: i32) -> i32 {
    let index = x + z * infoI32.grid_size_x;

    return select(-1, index, x >= 0 && x < infoI32.grid_size_x && z >= 0 && z < infoI32.grid_size_z);
}

@compute
@workgroup_size(wsz, wsx)
fn forward_diff(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    if (z + 1 < infoI32.grid_size_z) {
        dp_1_z[zx(z, x)] = (p_current[zx(z + 1, x)] - p_current[zx(z, x)]) / infoF32.dz;
    }
    if (x + 1 < infoI32.grid_size_x) {
        dp_1_x[zx(z, x)] = (p_current[zx(z, x + 1)] - p_current[zx(z, x)]) / infoF32.dx;
    }
}

@compute
@workgroup_size(wsz, wsx)
fn backward_diff(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    if (z - 1 >= 0) {
        dp_2_z[zx(z, x)] = (dp_1_z[zx(z, x)] - dp_1_z[zx(z - 1, x)]) / infoF32.dz;
    }
    if (x - 1 >= 0) {
        dp_2_x[zx(z, x)] = (dp_1_x[zx(z, x)] - dp_1_x[zx(z, x - 1)]) / infoF32.dx;
    }
}

@compute
@workgroup_size(wsz, wsx)
fn after_forward(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    if (is_z_absorption[zx(z, x)] == 1) {
        phi_z[zx(z, x)] = absorption_z[zx(z, x)] * phi_z[zx(z, x)] + (absorption_z[zx(z, x)] - 1) * dp_1_z[zx(z, x)];
        dp_1_z[zx(z, x)] += phi_z[zx(z, x)];
    }
    if (is_x_absorption[zx(z, x)] == 1) {
        phi_x[zx(z, x)] = absorption_x[zx(z, x)] * phi_x[zx(z, x)] + (absorption_x[zx(z, x)] - 1) * dp_1_x[zx(z, x)];
        dp_1_x[zx(z, x)] += phi_x[zx(z, x)];
    }
}

@compute
@workgroup_size(wsz, wsx)
fn after_backward(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    if (is_z_absorption[zx(z, x)] == 1) {
        psi_z[zx(z, x)] = absorption_z[zx(z, x)] * psi_z[zx(z, x)] + (absorption_z[zx(z, x)] - 1) * dp_2_z[zx(z, x)];
        dp_2_z[zx(z, x)] += psi_z[zx(z, x)];
    }
    if (is_x_absorption[zx(z, x)] == 1) {
        psi_x[zx(z, x)] = absorption_x[zx(z, x)] * psi_x[zx(z, x)] + (absorption_x[zx(z, x)] - 1) * dp_2_x[zx(z, x)];
        dp_2_x[zx(z, x)] += psi_x[zx(z, x)];
    }
}

@compute
@workgroup_size(wsz, wsx)
fn sim(@builtin(global_invocation_id) index: vec3<u32>) {
    let z: i32 = i32(index.x);
    let x: i32 = i32(index.y);

    p_next[zx(z, x)] = (c[zx(z, x)] * c[zx(z, x)]) * (dp_2_z[zx(z, x)] + dp_2_x[zx(z, x)]) * (infoF32.dt * infoF32.dt);

    p_next[zx(z, x)] += ((2. * p_current[zx(z, x)]) - p_previous[zx(z, x)]);

    for (var microphone_index: i32 = 0; microphone_index < infoI32.microphones_amount; microphone_index += 1)
    {
        if (z == microphone_z[microphone_index] && x == microphone_x[microphone_index])
        {
            //FLIPPED_MICROPHONES_SIM
        }
    }

    p_previous[zx(z, x)] = p_current[zx(z, x)];
    p_current[zx(z, x)] = p_next[zx(z, x)];
}

@compute
@workgroup_size(1)
fn incr_time() {
    infoI32.i += 1;
}
