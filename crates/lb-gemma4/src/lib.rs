//! Gemma 4 (gemma4 MoE, e.g. google gemma-4-26B-A4B-it) decoder loaded straight from GGUF,
//! checked against llama.cpp e14a779 (`src/models/gemma4.cpp`). See `docs/GEMMA4.md`.
//!
//! The expert machinery is lb-lfm2's (Route, Placement, PlacementStats, ExpertStore, Scraper,
//! Knobs, CapturedLayer, stage::rank), used unchanged, so the engine steps that change it
//! (dynamic cache, device router, prediction-guided keep, tiers) reach this model too.
pub mod config;
pub mod model;
pub mod moe;
pub mod numerics;
pub mod weights;

pub use config::Config;
pub use model::{Cache, Gemma4};
