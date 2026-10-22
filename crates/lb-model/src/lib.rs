//! BitNet b1.58 decoder in Burn with ternary linear layers.
pub mod model;
pub mod bench;
pub mod ops;
pub mod sample;
pub mod text;

pub use model::{BitNet, Config, KvCache};
