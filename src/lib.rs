// Re-export public components
pub use crate::bspline::BSpline;
pub use crate::univariate_function::UnivariateFunction;
pub use crate::kan_layer::KANLayer;
pub use crate::kan::KAN;

// Module declarations
pub mod bspline;
pub mod univariate_function;
pub mod kan_layer;
pub mod kan;

// Shader includes
pub mod shaders {
    pub const BSPLINE_SHADER: &str = include_str!("shaders/bspline.wgsl");
    pub const UNIVARIATE_FUNCTION_SHADER: &str = include_str!("shaders/univariate_function.wgsl");
    pub const WEIGHT_UPDATE_SHADER: &str = include_str!("shaders/weight_update.wgsl");
    pub const KAN_LAYER_SHADER: &str = include_str!("shaders/kan_layer.wgsl");
}
