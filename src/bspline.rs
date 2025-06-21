use std::ops::Range;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::*;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use bytemuck::{Pod, Zeroable};
use crate::shaders::BSPLINE_SHADER;

/// GPU-compatible representation of B-spline knots
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct KnotData {
    pub knot: f32,
}

/// GPU-compatible representation of B-spline parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BSplineParams {
    pub x: f32,
    pub degree: u32,
}

/// Represents a B-spline basis function for approximating univariate functions
pub struct BSpline {
    pub knots: Vec<f32>,
    pub degree: usize,
    // GPU resources
    pub device: Arc<Device>,  // Changed from Device
    pub queue: Arc<Queue>,    // Changed from Queue
    pub knots_buffer: Buffer,
    pub compute_pipeline: ComputePipeline,
    pub bind_group_layout: BindGroupLayout,
    pub output_buffer: Buffer,
}

impl BSpline {
    /// Create a new B-spline with uniformly spaced knots over the range
    pub async fn new(range: Range<f32>, num_knots: usize, degree: usize) -> Self {
        // Initialize WGPU
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");
            
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");
        
        // Create knots vector
        let mut knots = Vec::with_capacity(num_knots + 2 * degree);
        
        // Add padding knots at the beginning
        for _ in 0..degree {
            knots.push(range.start);
        }
        
        // Add interior knots
        let step = (range.end - range.start) / (num_knots as f32 - 1.0);
        for i in 0..num_knots {
            knots.push(range.start + i as f32 * step);
        }
        
        // Add padding knots at the end
        for _ in 0..degree {
            knots.push(range.end);
        }

        // Create knots buffer
        let knot_data: Vec<KnotData> = knots.iter().map(|&k| KnotData { knot: k }).collect();
        let knots_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Knots Buffer"),
            contents: bytemuck::cast_slice(&knot_data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        
        // Create output buffer
        let output_buffer_size = (knots.len() - degree - 1) * std::mem::size_of::<f32>();
        let output_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create shader module
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("BSpline Shader"),
            source: ShaderSource::Wgsl(BSPLINE_SHADER.into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("BSpline Bind Group Layout"),
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
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create compute pipeline
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("BSpline Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            compilation_options: PipelineCompilationOptions {
                constants: &HashMap::new(),
                zero_initialize_workgroup_memory: false,
            },
            label: Some("BSpline Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        
        BSpline {
            knots,
            degree,
            device: Arc::new(device),  // Wrap in Arc::new()
            queue: Arc::new(queue),    // Wrap in Arc::new()
            knots_buffer,
            compute_pipeline,
            bind_group_layout,
            output_buffer,
        }
    }
    
    /// Evaluate the B-spline basis functions at a given point (on GPU)
    pub async fn evaluate_basis_gpu(&self, x: f32) -> Vec<f32> {
        // Create params buffer with current x value
        let params = BSplineParams {
            x,
            degree: self.degree as u32,
        };
        
        let params_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        
        // Create bind group with updated params
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("BSpline Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.knots_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("BSpline Command Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("BSpline Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((self.knots.len() - self.degree - 1) as u32, 1, 1);
        }
        
        // Create staging buffer for reading results
        let output_size = (self.knots.len() - self.degree - 1) * std::mem::size_of::<f32>();
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy output buffer to staging buffer
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &staging_buffer,
            0,
            output_size as u64,
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
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        result
    }
    
/// Evaluate the B-spline basis functions at a given point (CPU fallback)
pub fn evaluate_basis(&self, x: f32) -> Vec<f32> {
    let num_basis = self.knots.len() - self.degree - 1;
    let mut basis = vec![0.0; num_basis];
    
    // Find the knot span where x lies
    let mut span = self.degree;
    for i in self.degree..(self.knots.len() - 1) {
        if x < self.knots[i + 1] {
            span = i;
            break;
        }
    }
    
    // Handle the case where x is exactly at the right boundary
    if x >= self.knots[self.knots.len() - 1] {
        span = self.knots.len() - self.degree - 2;
    }
    
    // Temporary array for the non-zero basis functions
    let mut N = vec![0.0; self.degree + 1];
    
    // Initialize the zeroth-degree basis function
    N[0] = 1.0;
    
    // Compute the basis functions using the Cox-de Boor recursion formula
    for j in 1..=self.degree {
        let mut saved = 0.0;
        for r in 0..j {
            let alpha = if self.knots[span + 1 + r] != self.knots[span + 1 + r - j] {
                (x - self.knots[span + 1 + r - j]) / (self.knots[span + 1 + r] - self.knots[span + 1 + r - j])
            } else {
                0.0
            };
            
            let temp = N[r];
            N[r] = saved + (1.0 - alpha) * temp;
            saved = alpha * temp;
        }
        N[j] = saved;
    }
    
    // Copy the non-zero basis functions to the output array
    for i in 0..=self.degree {
        let basis_idx = span - self.degree + i;
        if basis_idx < basis.len() {
            basis[basis_idx] = N[i];
        }
    }
    
    basis
}
}