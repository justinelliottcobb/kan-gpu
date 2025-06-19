# Kolmogorov-Arnold Networks with GPU Acceleration

This implementation provides a Rust implementation of Kolmogorov-Arnold Networks (KANs) with GPU acceleration using `wgpu`. KANs are a unique type of neural network architecture inspired by the Kolmogorov-Arnold representation theorem, which states that any multivariate continuous function can be represented as a composition of continuous functions of a single variable and addition operations.

## Key Components

### 1. B-Spline Basis Functions (`bspline.rs`)
- Represents the foundation for the learnable univariate functions
- Provides knot generation and basis function evaluation
- GPU-accelerated evaluation using the `bspline.wgsl` shader

### 2. Univariate Functions (`univariate_function.rs`)
- Implements learnable univariate functions using B-spline basis
- Each function has weights that are learned during training
- GPU-accelerated evaluation and weight updates using corresponding shaders

### 3. KAN Layer (`kan_layer.rs`)
- Combines multiple univariate functions
- Handles projections from inputs to function inputs
- Provides forward and backward pass implementations

### 4. Complete KAN (`kan.rs`)
- Manages multiple KAN layers
- Coordinates forward and backward passes through the network
- Implements training methods with backpropagation

### 5. WGSL Shaders
- `bspline.wgsl`: Evaluates B-spline basis functions on the GPU
- `univariate_function.wgsl`: Computes univariate function outputs
- `weight_update.wgsl`: Updates function weights during backpropagation
- `kan_layer.wgsl`: Manages the forward pass through a KAN layer

## GPU Acceleration

The implementation accelerates KAN computations by:
1. **Parallel Evaluation of Basis Functions**: Computing multiple basis function values simultaneously
2. **Efficient Function Evaluation**: Computing dot products and function outputs in parallel
3. **Parallel Weight Updates**: Updating multiple weights simultaneously during training

Each component includes both GPU-accelerated methods (`*_gpu`) and CPU fallback implementations for comparison and testing.

## Usage Example

A simple example that approximates the sine function is included in `main.rs`. The code:
1. Creates a KAN with specific dimensions
2. Generates training data (sine function samples)
3. Trains the network for several epochs
4. Tests the trained network on unseen inputs

## Next Steps for Improvement

This implementation could be extended in several ways:

1. **Full GPU Pipeline**: Currently, some data transfers between CPU and GPU could be eliminated by keeping more computations on the GPU.

2. **Improved Layer Coordination**: The coordination between layers could be improved for more efficient training.

3. **Higher-Order KANs**: Support for higher-order KANs where functions can compose into more complex structures.

4. **Optimized Memory Usage**: More efficient use of GPU memory and reduction of buffer creation overhead.

5. **Advanced B-Spline Implementations**: More efficient B-spline implementations that handle edge cases better.

6. **Performance Tuning**: Workgroup sizes and memory layouts could be optimized for specific GPU architectures.

## Running the Project

To run the project, use:

```bash
cargo run --release
```

The project requires a system with GPU support compatible with the `wgpu` crate. For systems without GPU support, the CPU fallback methods can be used by changing the method calls in `main.rs`.