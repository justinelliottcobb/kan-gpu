use std::ops::Range;
use wgpu::*;
use wgpu::util::BufferInitDescriptor;
use bytemuck::{Pod, Zeroable};
use crate::univariate_function::UnivariateFunction;
use crate::shaders::KAN_LAYER_SHADER;

/// GPU-compatible representation of KAN layer parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LayerParams {
    pub input_dim: u32,
    pub output_dim: u32,
    pub num_functions: u32,
    pub _padding: u32, // For alignment
}

/// A single layer in a Kolmogorov-Arnold Network (GPU accelerated)
pub struct KANLayer {
    pub input_dim: usize,
    pub output_dim: usize,
    pub functions: Vec<UnivariateFunction>,
    pub projection: Vec<f32>,
    // GPU resources
    pub device: Device,
    pub queue: Queue,
    pub projection_buffer: Buffer,
    pub compute_pipeline: ComputePipeline,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
    pub output_buffer: Buffer,
}

impl KANLayer {
    /// Create a new KAN layer
    pub async fn new(input_dim: usize, output_dim: usize, num_functions: usize, 
               range: Range<f32>, num_knots: usize, degree: usize) -> Self {
        // Create univariate functions
        let mut functions = Vec::with_capacity(num_functions);
        for _ in 0..num_functions {
            functions.push(UnivariateFunction::new(range.clone(), num_knots, degree).await);
        }
        
        // Initialize projection matrix with small random values
        let mut projection = Vec::with_capacity(input_dim * num_functions);
        for _ in 0..input_dim * num_functions {
            projection.push(rand::random::<f32>() * 0.1 - 0.05);
        }
        
        // Get device and queue reference from the first function
        let device = functions[0].device.clone();
        let queue = functions[0].queue.clone();
        
        // Create projection buffer
        let projection_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Projection Buffer"),
            contents: bytemuck::cast_slice(&projection),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        // Create output buffer
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Layer Output Buffer"),
            size: (output_dim * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create layer parameters
        let layer_params = LayerParams {
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            num_functions: num_functions as u32,
            _padding: 0, // For alignment
        };
        
        let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Layer Params Buffer"),
            contents: bytemuck::bytes_of(&layer_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        
        // Create shader module
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("KAN Layer Shader"),
            source: ShaderSource::Wgsl(KAN_LAYER_SHADER.into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Layer Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create compute pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Layer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Layer Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        
        // Create a dummy bind group - we'll create the real one in forward pass
        let dummy_input_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Dummy Input Buffer"),
            size: std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let dummy_params = LayerParams {
            input_dim: input_dim as u32,
            output_dim: output_dim as u32,
            num_functions: num_functions as u32,
            _padding: 0,
        };
        
        let dummy_params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Dummy Params Buffer"),
            contents: bytemuck::bytes_of(&dummy_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Dummy Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: dummy_params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: dummy_input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: projection_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        
        KANLayer {
            input_dim,
            output_dim,
            functions,
            projection,
            device,
            queue,
            projection_buffer,
            compute_pipeline,
            bind_group_layout,
            bind_group,
            output_buffer,
        }
    }
    
    /// Forward pass through the layer (GPU)
    pub async fn forward_gpu(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim, "Input dimension mismatch");
        
        // Create input buffer
        let input_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(input),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        // Create layer parameters
        let layer_params = LayerParams {
            input_dim: self.input_dim as u32,
            output_dim: self.output_dim as u32,
            num_functions: self.functions.len() as u32,
            _padding: 0, // For alignment
        };
        
        let params_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Layer Params Buffer"),
            contents: bytemuck::bytes_of(&layer_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        
        // Create bind group for this forward pass
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Layer Forward Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.projection_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: self.output_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Layer Forward Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Layer Forward Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(self.output_dim as u32, 1, 1);
        }
        
        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Layer Output Staging Buffer"),
            size: (self.output_dim * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy output buffer to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &staging_buffer,
            0,
            (self.output_dim * std::mem::size_of::<f32>()) as u64,
        );
        
        // Submit command buffer
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Read results
        let buffer_slice = staging_buffer.slice(..);
        
        // Wait for the GPU to finish
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(Maintain::Wait);
        
        let result = receiver.recv_async().await.unwrap().unwrap();
        let data = buffer_slice.get_mapped_range();
        let output: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        output
    }
    
    /// Forward pass through the layer (CPU fallback)
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim, "Input dimension mismatch");
        
        let mut output = vec![0.0; self.output_dim];
        
        // For each output dimension
        for o in 0..self.output_dim {
            // For each univariate function
            for (f_idx, function) in self.functions.iter().enumerate() {
                // Compute the projection
                let mut proj = 0.0;
                for i in 0..self.input_dim {
                    proj += input[i] * self.projection[i * self.functions.len() + f_idx];
                }
                
                // Evaluate the function and add to output
                output[o] += function.evaluate(proj);
            }
        }
        
        output
    }
    
    /// Backward pass to compute gradients (GPU)
    pub async fn backward_gpu(&mut self, input: &[f32], output_grad: &[f32], learning_rate: f32) {
        // This is a simplified backward pass implementation.
        // A full implementation would use GPU for all computations.
        
        // For each output dimension
        for o in 0..self.output_dim {
            // For each univariate function
            for (f_idx, function) in self.functions.iter_mut().enumerate() {
                // Compute the projection
                let mut proj = 0.0;
                for i in 0..self.input_dim {
                    proj += input[i] * self.projection[i * self.functions.len() + f_idx];
                }
                
                // Update function weights
                function.update_gpu(proj, output_grad[o], learning_rate).await;
                
                // Update projection weights (simplified - normally would be done on GPU)
                for i in 0..self.input_dim {
                    let idx = i * self.functions.len() + f_idx;
                    self.projection[idx] -= learning_rate * output_grad[o] * input[i];
                }
            }
        }
        
        // Update projection buffer with new weights
        self.queue.write_buffer(
            &self.projection_buffer,
            0,
            bytemuck::cast_slice(&self.projection),
        );
    }
    
    /// Backward pass to compute gradients (CPU fallback)
    pub fn backward(&mut self, input: &[f32], output_grad: &[f32], learning_rate: f32) {
        // For each output dimension
        for o in 0..self.output_dim {
            // For each univariate function
            for (f_idx, function) in self.functions.iter_mut().enumerate() {
                // Compute the projection
                let mut proj = 0.0;
                for i in 0..self.input_dim {
                    proj += input[i] * self.projection[i * self.functions.len() + f_idx];
                }
                
                // Update function weights
                function.update(proj, output_grad[o], learning_rate);
                
                // Update projection weights
                for i in 0..self.input_dim {
                    let idx = i * self.functions.len() + f_idx;
                    self.projection[idx] -= learning_rate * output_grad[o] * input[i];
                }
            }
        }
    }
}