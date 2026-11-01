//! Self-calibration for the MoE engine (docs/V2_PIPELINE.md):
//! measured timings of this box's cards and host (`timings`, produced by `lb-gpu-bench`),
//! the card topology from the drivers (`topology`), routing traces from NautiBurn scrapes
//! (`trace`), a discrete-event model of the layer pipeline (`sim`), and the tuner that
//! picks the staging config for this hardware and writes it for the engine (`tune`).
pub mod sim;
pub mod timings;
pub mod topology;
pub mod trace;
pub mod tune;
