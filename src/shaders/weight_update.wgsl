// Weight update shader for backpropagation

@group(0) @binding(0)
var<storage, read> basis: array<f32>;

struct UpdateParams {
    gradient: f32,
    learning_rate: f32,
};

@group(0) @binding(1)
var<uniform> params: UpdateParams;

@group(0) @binding(2)
var<storage, read_write> weights: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i < arrayLength(&weights) && i < arrayLength(&basis)) {
        // Apply gradient update rule: w = w - lr * grad * basis
        weights[i] -= params.learning_rate * params.gradient * basis[i];
    }
}
