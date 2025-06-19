use std::ops::Range;
use wgpu::*;
use crate::kan_layer::KANLayer;

/// A complete Kolmogorov-Arnold Network (GPU accelerated)
pub struct KAN {
    pub layers: Vec<KANLayer>,
    pub device: Device,
    pub queue: Queue,
}

impl KAN {
    /// Create a new KAN with specified layer dimensions
    pub async fn new(layer_dims: &[usize], functions_per_layer: &[usize], 
               range: Range<f32>, num_knots: usize, degree: usize) -> Self {
        assert!(layer_dims.len() >= 2, "Need at least input and output dimensions");
        assert_eq!(layer_dims.len() - 1, functions_per_layer.len(), "Need functions_per_layer for each layer");
        
        let mut layers = Vec::with_capacity(layer_dims.len() - 1);
        
        for i in 0..layer_dims.len() - 1 {
            layers.push(KANLayer::new(
                layer_dims[i],
                layer_dims[i + 1],
                functions_per_layer[i],
                range.clone(),
                num_knots,
                degree
            ).await);
        }
        
        // Get device and queue from the first layer
        let device = layers[0].device.clone();
        let queue = layers[0].queue.clone();
        
        KAN { layers, device, queue }
    }
    
    /// Forward pass through the network (GPU)
    pub async fn forward_gpu(&self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        
        for layer in &self.layers {
            current = layer.forward_gpu(&current).await;
        }
        
        current
    }
    
    /// Forward pass through the network (CPU fallback)
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        
        current
    }
    
    /// Train the network using backpropagation (GPU)
    pub async fn train_gpu(&mut self, input: &[f32], target: &[f32], learning_rate: f32) -> f32 {
        // Forward pass
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_vec());
        
        for layer in &self.layers {
            activations.push(layer.forward_gpu(activations.last().unwrap()).await);
        }
        
        // Compute loss
        let output = activations.last().unwrap();
        let mut loss = 0.0;
        let mut output_grad = output.clone();
        
        for i in 0..output.len() {
            let error = output[i] - target[i];
            loss += 0.5 * error * error;
            output_grad[i] = error;  // Gradient of MSE loss
        }
        
        // Backward pass
        for i in (0..self.layers.len()).rev() {
            let input_activation = &activations[i];
            self.layers[i].backward_gpu(input_activation, &output_grad, learning_rate).await;
            
            // Compute gradients for the previous layer
            if i > 0 {
                // This is a simplified version - a real implementation would need to properly
                // compute gradients through the univariate functions
                output_grad = vec![0.0; activations[i].len()];
            }
        }
        
        loss
    }
    
    /// Train the network using backpropagation (CPU fallback)
    pub fn train(&mut self, input: &[f32], target: &[f32], learning_rate: f32) -> f32 {
        // Forward pass
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_vec());
        
        for layer in &self.layers {
            activations.push(layer.forward(activations.last().unwrap()));
        }
        
        // Compute loss
        let output = activations.last().unwrap();
        let mut loss = 0.0;
        let mut output_grad = output.clone();
        
        for i in 0..output.len() {
            let error = output[i] - target[i];
            loss += 0.5 * error * error;
            output_grad[i] = error;  // Gradient of MSE loss
        }
        
        // Backward pass
        for i in (0..self.layers.len()).rev() {
            let input_activation = &activations[i];
            self.layers[i].backward(input_activation, &output_grad, learning_rate);
            
            // Compute gradients for the previous layer
            if i > 0 {
                // This is a simplified version - a real implementation would need to properly
                // compute gradients through the univariate functions
                output_grad = vec![0.0; activations[i].len()];
            }
        }
        
        loss
    }
}
