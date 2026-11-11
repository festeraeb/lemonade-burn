//! Work goes to the weights: an expert's weights never move after load. An **expert host**
//! holds a set of `(layer, expert)` and computes them; the **trunk** (embedding, attention /
//! conv, norms, router, head: `lb-lfm2`) sends each MoE layer's activation to the hosts that
//! hold the routed experts and gets the expert outputs back, then does the weighted
//! ascending-expert-id sum itself. See `docs/WORK_TO_WEIGHTS.md`.
//!
//! * [`wire`]: the frame format (request: layer, ids, row groups, activations; reply: one
//!   output block per expert, unweighted).
//! * [`transport`]: in-process, TCP, unix socket and (feature `rdma`) RoCE queue pair.
//! * [`placement`]: placement as data, one JSON file for trunk and hosts.
//! * [`model`]: the expert tensors of a GGUF, nothing else of the model.
//! * [`expert`]: the CPU reference arithmetic (shared with the trunk's one-process path).
//! * [`host`]: the host and its backends (CPU; device through `lb-q4-gpu` with `gpu`/`cuda`).
//! * [`client`]: the trunk's end of a connection.
//! * [`testgguf`]: a tiny synthetic MoE GGUF for tests.

pub mod client;
pub mod expert;
pub mod host;
pub mod model;
pub mod placement;
#[cfg(feature = "rdma")]
pub mod rdma;
pub mod testgguf;
pub mod transport;
pub mod wire;

pub use client::HostClient;
pub use host::{Backend, CpuBackend, Host};
pub use model::ModelExperts;
pub use placement::{HeldSet, HostSpec, PlacementMap, Wait, LOCAL};
pub use wire::{HostInfo, RunReply};
