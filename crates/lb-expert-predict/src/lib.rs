//! Offline MoE routing analysis for lemonade-burn (`docs/EXPERT_PREDICT.md`).
//!
//! * [`metrics`]: the router maths shared by the analysis (RMS norm, sigmoid router scores,
//!   top-k with the confidence margin) and the set-overlap scores of a predicted expert set
//!   against the one the engine actually chose (recall@k, Jaccard, all-k hit).
//! * [`capture`]: the on-disk format the capture runner writes, chunk by chunk, so a run can
//!   stop anywhere and resume.
//! * [`standin`]: scores one expert's outputs as a stand-in for another's on the hidden
//!   states that were routed to the latter (cosine, relative L2), and routing concentration.
pub mod capture;
pub mod metrics;
pub mod standin;
