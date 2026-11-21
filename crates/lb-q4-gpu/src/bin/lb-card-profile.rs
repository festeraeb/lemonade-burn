//! `lb-card-profile [--full]`: for each card this process can see, which card profile matched and every knob's value and source
//! (default / profile / env). Probes only (driver queries, adapter enumeration, sysfs); no device or context is created, so it is safe
//! next to a running seat. `LB_CARD_PROFILE_DIR=/path` adds or replaces profiles from JSON files without a rebuild; env vars override.
fn main() {
    print!("{}", lb_q4_gpu::store::card_profile_report_opts(std::env::args().any(|a| a == "--full")));
}
