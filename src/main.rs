use kan_gpu::{KAN};
use std::ops::Range;

#[tokio::main]
async fn main() {
    println!("Initializing KAN with GPU acceleration...");
    
    // Create a simple KAN for approximating sin(x)
    let mut kan = KAN::new(
        &[1, 10, 1],         // Input dim, hidden dim, output dim
        &[5, 5],             // Functions per layer
        -5.0..5.0,           // Range for the univariate functions
        10,                  // Number of knots for B-splines
        3                    // Degree of B-splines
    ).await;
    
    // Training data: sin(x) for x in [-π, π]
    let num_samples = 100;
    let mut inputs = Vec::with_capacity(num_samples);
    let mut targets = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let x = -std::f32::consts::PI + 2.0 * std::f32::consts::PI * i as f32 / (num_samples as f32 - 1.0);
        inputs.push(vec![x]);
        targets.push(vec![x.sin()]);
    }
    
    // Train the KAN
    let num_epochs = 1000;
    let learning_rate = 0.01;
    
    println!("Starting training for {} epochs...", num_epochs);
    
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        
        for i in 0..num_samples {
            total_loss += kan.train_gpu(&inputs[i], &targets[i], learning_rate).await;
        }
        
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {}", epoch, total_loss / num_samples as f32);
        }
    }
    
    println!("Training complete! Testing the model...");
    
    // Test the trained KAN
    for i in 0..10 {
        let x = -std::f32::consts::PI + 2.0 * std::f32::consts::PI * i as f32 / 9.0;
        let input = vec![x];
        let output = kan.forward_gpu(&input).await;
        
        println!("sin({:.4}) = {:.6} (KAN prediction: {:.6})", x, x.sin(), output[0]);
    }
}
