//! LFM2 / LFM2-MoE (LiquidAI) decoder on Burn, loaded straight from GGUF.
//! See `docs/LFM2_CUBECL.md` for the architecture, kernels, placement and verification.
pub mod cache;
pub mod config;
pub mod dense;
#[cfg(feature = "gpu")]
pub mod device_decode;
#[cfg(feature = "gpu")]
pub mod device_prefill;
#[cfg(feature = "gpu")]
pub mod inbox;
pub mod knobs;
pub mod model;
pub mod moe;
pub mod remote;
pub mod stage;
pub mod weights;

pub use config::Config;
pub use knobs::Knobs;
pub use model::{Cache, CapturedLayer, Lfm2};
