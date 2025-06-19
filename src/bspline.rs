use std::ops::Range;
use wgpu::*;
use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;
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
    pub device: Device,
    pub queue: Queue,
    pub knots_buffer: Buffer,
    pub compute_pipeline: ComputePipeline,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
    pub output_buffer: Buffer,
}

impl BSpline {
    /// Create a new B-spline with uniformly spaced knots over the range
    pub async fn new(&self, range: Range<f32>, num_knots: usize, degree: usize) -> Self {
        // Initialize WGPU
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version: Gles3MinorVersion::default(),
            flags: InstanceFlags::default(),
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
            label: Some("BSpline Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        
        // Create params buffer (will be updated per evaluation)
        let params = BSplineParams {
            x: 0.0, // Will be updated later
            degree: degree as u32,
        };
        
        let params_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });
        
        // Create bind group
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("BSpline Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: knots_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        BSpline {
            knots,
            degree,
            device,
            queue,
            knots_buffer,
            compute_pipeline,
            bind_group_layout,
            bind_group,
            output_buffer,
        }
    }
    
    /// Evaluate the B-spline basis functions at a given point (on GPU)
    pub async fn evaluate_basis_gpu(&self, x: f32) -> Vec<f32> {
        // Update params buffer with new x value
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
            timestamp_writes: None,
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
        let mut basis = vec![0.0; self.knots.len() - self.degree - 1];
        
        // Find the knot span
        let mut span = self.degree;
        for i in self.degree..self.knots.len() - 1 {
            if x < self.knots[i + 1] {
                span = i;
                break;
            }
        }
        
        // Initialize degree 0 basis functions
        let mut N = vec![0.0; self.degree + 1];
        for i in 0..=self.degree {
            if x >= self.knots[span - i] && x < self.knots[span - i + 1] {
                N[i] = 1.0;
            }
        }
        
        // Build up basis functions of increasing degree
        for d in 1..=self.degree {
            for i in 0..=(self.degree - d) {
                let left = if self.knots[span - i + d] - self.knots[span - i] > 0.0 {
                    (x - self.knots[span - i]) / (self.knots[span - i + d] - self.knots[span - i])
                } else {
                    0.0
                };
                
                let right = if self.knots[span - i + d + 1] - self.knots[span - i + 1] > 0.0 {
                    (self.knots[span - i + d + 1] - x) / (self.knots[span - i + d + 1] - self.knots[span - i + 1])
                } else {
                    0.0
                };
                
                N[i] = left * N[i] + right * N[i + 1];
            }
        }
        
        // Fill the basis vector with non-zero values
        for i in 0..=self.degree {
            if span - self.degree + i < basis.len() {
                basis[span - self.degree + i] = N[i];
            }
        }
        
        basis
    }
}