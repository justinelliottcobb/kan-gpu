// B-spline shader for evaluating basis functions
@group(0) @binding(0)
var<storage, read> knots: array<f32>;

@group(0) @binding(1)
var<storage, read_write> basis_output: array<f32>;

struct Params {
    x: f32,
    degree: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

// Recursive implementation of B-spline basis function
fn evaluate_basis_recursive(x: f32, degree: u32, i: u32) -> f32 {
    // Base case: degree 0
    if (degree == 0u) {
        if (x >= knots[i] && x < knots[i + 1u]) || (x == knots[i + 1u] && i + 1u == arrayLength(&knots) - 1u) {
            return 1.0;
        }
        return 0.0;
    }
    
    // Recursive case
    var result: f32 = 0.0;
    
    // First term
    let left_denom = knots[i + degree] - knots[i];
    if (left_denom > 0.0) {
        result += ((x - knots[i]) / left_denom) * evaluate_basis_recursive(x, degree - 1u, i);
    }
    
    // Second term
    let right_denom = knots[i + degree + 1u] - knots[i + 1u];
    if (right_denom > 0.0) {
        result += ((knots[i + degree + 1u] - x) / right_denom) * evaluate_basis_recursive(x, degree - 1u, i + 1u);
    }
    
    return result;
}

// Non-recursive implementation using dynamic programming
fn evaluate_basis_dp(x: f32, degree: u32, idx: u32) -> f32 {
    var N: array<f32, 10>; // Assuming max degree is 9
    
    // Find knot span
    var span: u32 = degree;
    for (var i: u32 = degree; i < arrayLength(&knots) - 1u; i = i + 1u) {
        if (x < knots[i + 1u]) {
            span = i;
            break;
        }
    }
    
    // Initialize degree 0 basis functions
    for (var i: u32 = 0u; i <= degree; i = i + 1u) {
        if (x >= knots[span - i] && x < knots[span - i + 1u]) {
            N[i] = 1.0;
        } else {
            N[i] = 0.0;
        }
    }
    
    // Build up basis functions of increasing degree
    for (var d: u32 = 1u; d <= degree; d = d + 1u) {
        for (var i: u32 = 0u; i <= degree - d; i = i + 1u) {
            var left: f32 = 0.0;
            var right: f32 = 0.0;
            
            if (knots[span - i + d] - knots[span - i] > 0.0) {
                left = (x - knots[span - i]) / (knots[span - i + d] - knots[span - i]) * N[i];
            }
            
            if (knots[span - i + d + 1u] - knots[span - i + 1u] > 0.0) {
                right = (knots[span - i + d + 1u] - x) / (knots[span - i + d + 1u] - knots[span - i + 1u]) * N[i + 1u];
            }
            
            N[i] = left + right;
        }
    }
    
    // Find which basis function corresponds to our index
    let basis_idx = span - degree + idx;
    if (basis_idx < arrayLength(&basis_output)) {
        return N[idx];
    }
    
    return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let x = params.x;
    let degree = params.degree;
    
    if (idx < arrayLength(&basis_output)) {
        // Use the more efficient dynamic programming version
        basis_output[idx] = evaluate_basis_dp(x, degree, idx);
    }
}
