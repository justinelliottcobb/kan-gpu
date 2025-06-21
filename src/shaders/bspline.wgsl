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

// Implementation using Cox-de Boor recursion (matches CPU implementation)
fn evaluate_basis_cox_deboor(x: f32, degree: u32, target_idx: u32) -> f32 {
    let num_basis = arrayLength(&basis_output);
    
    // Handle out of bounds
    if (target_idx >= num_basis) {
        return 0.0;
    }
    
    // Find the knot span where x lies
    var span: u32 = degree;
    for (var i: u32 = degree; i < arrayLength(&knots) - 1u; i = i + 1u) {
        if (x < knots[i + 1u]) {
            span = i;
            break;
        }
    }
    
    // Handle the case where x is exactly at the right boundary
    if (x >= knots[arrayLength(&knots) - 1u]) {
        span = arrayLength(&knots) - degree - 2u;
    }
    
    // Check if this target_idx could be non-zero for this span
    let first_nonzero = span - degree;
    let last_nonzero = span;
    
    if (target_idx < first_nonzero || target_idx > last_nonzero) {
        return 0.0;
    }
    
    // Local index within the non-zero range
    let local_idx = target_idx - first_nonzero;
    
    // Temporary array for the non-zero basis functions
    var N: array<f32, 10>; // Support up to degree 9
    
    // Initialize the zeroth-degree basis function
    for (var i: u32 = 0u; i <= degree; i = i + 1u) {
        N[i] = 0.0;
    }
    N[0] = 1.0;
    
    // Compute the basis functions using the Cox-de Boor recursion formula
    for (var j: u32 = 1u; j <= degree; j = j + 1u) {
        var saved: f32 = 0.0;
        for (var r: u32 = 0u; r < j; r = r + 1u) {
            let knot_left = span + 1u + r - j;
            let knot_right = span + 1u + r;
            
            var alpha: f32 = 0.0;
            if (knot_left < arrayLength(&knots) && knot_right < arrayLength(&knots) && 
                knots[knot_right] != knots[knot_left]) {
                alpha = (x - knots[knot_left]) / (knots[knot_right] - knots[knot_left]);
            }
            
            let temp = N[r];
            N[r] = saved + (1.0 - alpha) * temp;
            saved = alpha * temp;
        }
        N[j] = saved;
    }
    
    // Return the appropriate basis function value
    if (local_idx <= degree) {
        return N[local_idx];
    }
    
    return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let x = params.x;
    let degree = params.degree;
    
    if (idx < arrayLength(&basis_output)) {
        basis_output[idx] = evaluate_basis_cox_deboor(x, degree, idx);
    }
}