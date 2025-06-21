use kan_gpu::BSpline;
use std::ops::Range;

#[tokio::main]
async fn main() {
    env_logger::init();
    
    println!("Testing B-spline basis function evaluation...");
    
    // Create a simple B-spline
    let bspline = BSpline::new(-1.0..1.0, 5, 3).await;
    
    println!("B-spline info:");
    println!("  Knots: {:?}", bspline.knots);
    println!("  Degree: {}", bspline.degree);
    println!("  Expected basis functions: {}", bspline.knots.len() - bspline.degree - 1);
    
    // Test evaluation at several points
    let test_points = [-0.5, 0.0, 0.5];
    
    for &x in &test_points {
        println!("\nEvaluating at x = {}", x);
        
        // CPU evaluation
        let cpu_result = bspline.evaluate_basis(x);
        println!("CPU result: {:?}", cpu_result);
        
        // Check for any non-zero values
        let cpu_nonzero: Vec<(usize, f32)> = cpu_result.iter().enumerate()
            .filter(|(_, &val)| val.abs() > 1e-6)
            .map(|(i, &val)| (i, val))
            .collect();
        println!("CPU non-zero: {:?}", cpu_nonzero);
        
        // GPU evaluation
        let gpu_result = bspline.evaluate_basis_gpu(x).await;
        println!("GPU result: {:?}", gpu_result);
        
        // Check for any non-zero values
        let gpu_nonzero: Vec<(usize, f32)> = gpu_result.iter().enumerate()
            .filter(|(_, &val)| val.abs() > 1e-6)
            .map(|(i, &val)| (i, val))
            .collect();
        println!("GPU non-zero: {:?}", gpu_nonzero);
        
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