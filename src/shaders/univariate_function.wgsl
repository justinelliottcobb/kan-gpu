// Univariate function shader for evaluating the function

@group(0) @binding(0)
var<storage, read> basis: array<f32>;

@group(0) @binding(1)
var<storage, read> weights: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: f32;

@compute @workgroup_size(1)
fn main() {
    var result: f32 = 0.0;
    
    // Compute the weighted sum of basis functions
    for (var i: u32 = 0u; i < arrayLength(&basis); i = i + 1u) {
        if (i < arrayLength(&weights)) {
            result += basis[i] * weights[i];
        }
    }
    
    // Store result
    output = result;
}
