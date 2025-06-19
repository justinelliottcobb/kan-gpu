use kan_gpu::BSpline;
use std::ops::Range;

#[tokio::main]
async fn main() {
    env_logger::init();
    
    println!("Testing B-spline basis function evaluation...");
    
    // Create a simple B-spline
    let bspline = BSpline::new(-1.0..1.0, 5, 3).await;
    
    // Test evaluation at several points
    let test_points = [-0.5, 0.0, 0.5];
    
    for &x in &test_points {
        println!("\nEvaluating at x = {}", x);
        
        // CPU evaluation
        let cpu_result = bspline.evaluate_basis(x);
        println!("CPU result: {:?}", cpu_result);
        
        // GPU evaluation
        let gpu_result = bspline.evaluate_basis_gpu(x).await;
        println!("GPU result: {:?}", gpu_result);
        
        // Compare results
        let max_diff = cpu_result.iter()
            .zip(gpu_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        
        println!("Max difference: {}", max_diff);
        
        if max_diff < 1e-5 {
            println!("✓ Results match within tolerance");
        } else {
            println!("✗ Results differ significantly");
        }
    }
}
// This example tests the B-spline basis function evaluation on both CPU and GPU.
// It initializes a B-spline, evaluates it at several points, and compares the results.
// The expected output is that the results from CPU and GPU evaluations should match closely.
// If the maximum difference is within a small tolerance, it confirms that the GPU implementation is working correctly.
// Make sure to run this example in an environment where the kan_gpu crate is properly set up.
