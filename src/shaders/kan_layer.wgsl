// KAN Layer shader for forward pass

struct LayerParams {
    input_dim: u32,
    output_dim: u32,
    num_functions: u32,
    _padding: u32,
};

@group(0) @binding(0)
var<uniform> params: LayerParams;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read> projection: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let o = global_id.x;
    
    if (o < params.output_dim) {
        // Initialize output to zero
        output[o] = 0.0;
        
        // For each univariate function
        for (var f: u32 = 0u; f < params.num_functions; f = f + 1u) {
            var proj: f32 = 0.0;
            
            // Compute the projection
            for (var i: u32 = 0u; i < params.input_dim; i = i + 1u) {
                let proj_idx = i * params.num_functions + f;
                if (proj_idx < arrayLength(&projection) && i < arrayLength(&input)) {
                    proj += input[i] * projection[proj_idx];
                }
            }
            
            // In a real implementation, we would evaluate the univariate function here
            // However, for this shader, we're just using the projection directly
            // This is a simplification; in practice you'd need to coordinate between
            // this shader and the univariate function shader
            output[o] += proj;
        }
    }
}
