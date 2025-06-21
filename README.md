# Kolmogorov-Arnold Networks with GPU Acceleration

A Rust implementation of Kolmogorov-Arnold Networks (KANs) with GPU acceleration using WebGPU. This project explores KANs as an alternative to traditional Multi-Layer Perceptrons (MLPs), featuring learnable univariate functions on edges instead of fixed activation functions on nodes.

## 🎯 Project Status

- ✅ **B-spline Basis Functions**: Working CPU and GPU implementations with perfect agreement
- ✅ **GPU Infrastructure**: Complete WebGPU setup with compute shaders
- ✅ **Univariate Functions**: Learnable B-spline based functions (CPU + GPU)
- 🚧 **KAN Layers**: Forward and backward pass implementations
- 🚧 **Complete KAN**: Multi-layer networks with training
- 🚧 **Training Examples**: Learning simple functions like sin(x)

## 🧠 What are Kolmogorov-Arnold Networks?

KANs are inspired by the **Kolmogorov-Arnold representation theorem**, which states that any multivariate continuous function can be represented as a composition of continuous functions of one variable and addition operations.

**Key Differences from MLPs:**
- **MLPs**: Fixed activation functions on nodes, learnable weights on edges
- **KANs**: Learnable activation functions on edges, no traditional weights

**Advantages:**
- More interpretable (can visualize learned functions)
- Potentially more parameter-efficient
- Better at capturing complex mathematical relationships
- Natural sparsity through function pruning

## 🏗️ Architecture

```
src/
├── main.rs              # Example: training KAN to approximate sin(x)
├── lib.rs               # Public API and module exports
├── bspline.rs           # B-spline basis function evaluation
├── univariate_function.rs # Learnable univariate functions
├── kan_layer.rs         # Single KAN layer implementation
├── kan.rs               # Complete multi-layer KAN
└── shaders/             # WebGPU compute shaders
    ├── bspline.wgsl     # B-spline evaluation on GPU
    ├── univariate_function.wgsl # Function evaluation
    ├── weight_update.wgsl # Gradient updates
    └── kan_layer.wgsl   # Layer forward pass
```

## 🔬 Mathematical Foundation

### B-spline Basis Functions

KANs use B-spline basis functions to represent learnable univariate functions:

```
f(x) = Σ wᵢ * Bᵢ(x)
```

Where:
- `Bᵢ(x)` are B-spline basis functions (computed via Cox-de Boor recursion)
- `wᵢ` are learnable weights
- Each function has local support (only affects nearby input values)

### Cox-de Boor Recursion

The foundation of our B-spline evaluation:

**Base case (degree 0):**
```
N_{i,0}(x) = 1 if tᵢ ≤ x < tᵢ₊₁, otherwise 0
```

**Recursive case (degree p > 0):**
```
N_{i,p}(x) = (x - tᵢ)/(tᵢ₊ₚ - tᵢ) * N_{i,p-1}(x) + (tᵢ₊ₚ₊₁ - x)/(tᵢ₊ₚ₊₁ - tᵢ₊₁) * N_{i+1,p-1}(x)
```

## 🚀 Getting Started

### Prerequisites

- Rust 1.70+ with Cargo
- GPU with WebGPU support (most modern GPUs)

### Installation

```bash
git clone <repository-url>
cd kan_gpu
cargo build --release
```

### Running Examples

**Test B-spline basis functions:**
```bash
cargo run --example simple_test
```

**Train KAN on sin(x):**
```bash
cargo run --bin main
```

## 🛠️ GPU Acceleration

This implementation leverages GPU compute shaders for:

1. **Parallel B-spline Evaluation**: Computing multiple basis functions simultaneously
2. **Batch Function Evaluation**: Evaluating univariate functions for multiple inputs
3. **Parallel Weight Updates**: Updating function weights during backpropagation
4. **Memory Efficiency**: Sharing GPU resources using `Arc<Device>` and `Arc<Queue>`

### CPU vs GPU Performance

Both implementations are provided:
- **CPU**: Reference implementation for verification and debugging
- **GPU**: High-performance implementation for training and inference

The implementations are mathematically identical and produce the same results (verified by our test suite).

## 📊 Example Output

```
Testing B-spline basis function evaluation...
B-spline info:
  Knots: [-1.0, -1.0, -1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
  Degree: 3
  Expected basis functions: 7

Evaluating at x = 0
CPU result: [0.0, 0.0, 0.16666666, 0.6666666, 0.16666667, 0.0, 0.0]
GPU result: [0.0, 0.0, 0.16666666, 0.6666666, 0.16666667, 0.0, 0.0]
Max difference: 0
✓ Results match within tolerance
```

## 🔧 Dependencies

```toml
wgpu = "0.20"                    # WebGPU implementation
bytemuck = { version = "1.14", features = ["derive"] }  # Safe byte casting
tokio = { version = "1.34", features = ["full"] }       # Async runtime
flume = "0.11"                   # Async channels
rand = "0.8"                     # Random number generation
env_logger = "0.10"              # Logging for debugging
```

## 📚 Learning Resources

- **Original KAN Paper**: [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
- **B-spline Theory**: Carl de Boor's "A Practical Guide to Splines"
- **Cox-de Boor Algorithm**: Developed independently by Maurice Cox (1971) and Carl de Boor (1972)
- **WebGPU**: [wgpu-rs documentation](https://docs.rs/wgpu/)

## 🎓 Educational Value

This project demonstrates:

1. **Mathematical Implementation**: Translating theory (Kolmogorov-Arnold theorem) into working code
2. **GPU Programming**: Using compute shaders for mathematical operations
3. **Numerical Algorithms**: Cox-de Boor recursion for stable B-spline evaluation
4. **Software Architecture**: Clean separation between mathematical concepts and implementation details
5. **Performance Engineering**: Comparing CPU vs GPU implementations

## 🚧 Current Limitations

1. **Shader Constraints**: No recursion in WGSL requires iterative algorithms
2. **Memory Management**: GPU memory usage could be optimized further  
3. **Scalability**: Currently tested on small examples
4. **Features**: Missing some advanced KAN features (pruning, symbolic regression)

## 🔮 Future Work

- [ ] Complete training pipeline with backpropagation
- [ ] Benchmark against traditional MLPs
- [ ] Implement function pruning and sparsity
- [ ] Add symbolic regression capabilities
- [ ] Optimize memory usage and performance
- [ ] Support for higher-dimensional problems

## 🤝 Contributing

This is an educational project! Areas for contribution:

- Performance optimizations
- Additional test cases
- Documentation improvements
- Advanced KAN features
- Comparative studies with MLPs

## 📜 License

This project is for educational purposes. Please cite the original KAN paper if using concepts in research.

## 🙏 Acknowledgments

- MIT team for the original KAN research
- Carl de Boor for the foundational B-spline algorithms
- The Rust and WebGPU communities for excellent tools
- Walter Rudin for mathematical foundations (even if he didn't cover B-splines!)

---

*"The best way to understand something is to implement it from scratch."* - This project embodies that philosophy for Kolmogorov-Arnold Networks.