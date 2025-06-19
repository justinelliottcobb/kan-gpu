use std::ops::Range;
use wgpu::*;
use wgpu::util::BufferInitDescriptor;
use bytemuck::{Pod, Zeroable};
use crate::bspline::BSpline;
use crate::shaders::{UNIVARIATE_FUNCTION_SHADER, WEIGHT_UPDATE_SHADER};

/// GPU-compatible representation of update parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct UpdateParams {
    pub gradient: f32,
    pub learning_rate: f32,
}

/// Represents a learnable univariate function using B-spline basis (GPU accelerated)
pub struct UnivariateFunction {
    pub spline: BSpline,
    pub weights: Vec<f32>,
    // GPU resources
    pub device: Device,
    pub queue: Queue,
    pub weights_buffer: Buffer,
    pub compute_pipeline: ComputePipeline,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
    pub output_buffer: Buffer,
}

impl UnivariateFunction {
    /// Create a new univariate function with random weights
    pub async fn new(&self, range: Range<f32>, num_knots: usize, degree: usize) -> Self {
        let spline = BSpline::new(range, num_knots, degree).await;
        let num_weights = num_knots + degree - 1;
        
        // Initialize weights with small random values
        let mut weights = Vec::with_capacity(num_weights);
        for _ in 0..num_weights {
            weights.push(rand::random::<f32>() * 0.1 - 0.05);
        }
        
        // Get device and queue from the spline
        let device = &spline.device;
        let queue = &spline.queue;
        
        // Create weights buffer
        let weights_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Weights Buffer"),
            contents: bytemuck::cast_slice(&weights),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        // Create output buffer
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Function Output Buffer"),
            size: std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create shader module
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Univariate Function Shader"),
            source: ShaderSource::Wgsl(UNIVARIATE_FUNCTION_SHADER.into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Function Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
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
            label: Some("Function Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Function Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        
        // Create bind group
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Function Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: spline.output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: weights_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        
        UnivariateFunction { 
            spline, 
            weights, 
            device: device.clone(), 
            queue: queue.clone(),
            weights_buffer,
            compute_pipeline,
            bind_group_layout,
            bind_group,
            output_buffer
        }
    }
    
    /// Evaluate the function at a given point (GPU)
    pub async fn evaluate_gpu(&self, x: f32) -> f32 {
        // First evaluate the B-spline basis at x
        self.spline.evaluate_basis_gpu(x).await;
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Function Command Encoder"),
            timestamp_writes: None,
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Function Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }
        
        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Function Staging Buffer"),
            size: std::mem::size_of::<f32>() as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy output buffer to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of::<f32>() as u64,
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
        let result: f32 = *bytemuck::from_bytes(&data);
        
        drop(data);
        staging_buffer.unmap();
        
        result
    }
    
    /// Evaluate the function at a given point (CPU fallback)
    pub fn evaluate(&self, x: f32) -> f32 {
        let basis = self.spline.evaluate_basis(x);
        basis.iter().zip(self.weights.iter())
            .map(|(b, w)| b * w)
            .sum()
    }
    
    /// Update weights using gradients and learning rate (GPU)
    pub async fn update_gpu(&mut self, x: f32, gradient: f32, learning_rate: f32) {
        let basis = self.spline.evaluate_basis_gpu(x).await;
        
        // Create buffer with gradient and learning rate
        let update_params = UpdateParams {
            gradient,
            learning_rate,
        };
        
        let update_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Update Buffer"),
            contents: bytemuck::bytes_of(&update_params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        
        // Create shader for updating weights
        let update_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Weight Update Shader"),
            source: ShaderSource::Wgsl(WEIGHT_UPDATE_SHADER.into()),
        });
        
        // Create basis buffer
        let basis_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Basis Buffer"),
            contents: bytemuck::cast_slice(&basis),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        // Create bind group layout for update
        let update_bind_group_layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Update Bind Group Layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
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
        
        // Create bind group for update
        let update_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Update Bind Group"),
            layout: &update_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: basis_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: update_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: self.weights_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create pipeline layout for update
        let update_pipeline_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Update Pipeline Layout"),
            bind_group_layouts: &[&update_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipeline for update
        let update_pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Update Compute Pipeline"),
            layout: Some(&update_pipeline_layout),
            module: &update_shader,
            entry_point: "main",
        });
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Update Command Encoder"),
            timestamp_writes: None,
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Update Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&update_pipeline);
            compute_pass.set_bind_group(0, &update_bind_group, &[]);
            compute_pass.dispatch_workgroups(self.weights.len() as u32, 1, 1);
        }
        
        // Submit command buffer
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Also update CPU-side weights
        // Read updated weights from GPU
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Weights Staging Buffer"),
            size: (self.weights.len() * std::mem::size_of::<f32>()) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create command encoder to copy weights
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Weights Copy Encoder"),
            timestamp_writes: None,
        });
        
        encoder.copy_buffer_to_buffer(
            &self.weights_buffer,
            0,
            &staging_buffer,
            0,
            (self.weights.len() * std::mem::size_of::<f32>()) as u64,
        );
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Read results
        let buffer_slice = staging_buffer.slice(..);
        
        // Wait for the GPU to finish
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(Maintain::Wait);
        
        let result = receiver.recv_async().await.unwrap().unwrap();
        let data = buffer_slice.get_mapped_range();
        let updated_weights: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        // Update CPU-side weights
        self.weights = updated_weights;
    }
    
    /// Update weights using gradients and learning rate (CPU fallback)
    pub fn update(&mut self, x: f32, gradient: f32, learning_rate: f32) {
        let basis = self.spline.evaluate_basis(x);
        for (i, b) in basis.iter().enumerate() {
            if i < self.weights.len() {
                self.weights[i] -= learning_rate * gradient * b;
            }
        }
    }
}