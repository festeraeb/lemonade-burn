//! Print every Vulkan adapter with the kernel's facts about its card (family, BAR 0 vs VRAM, power cap, link, the engine's batch limit).
//! Enumerates adapters only: no device, queue or buffer is created, so it is safe next to a running seat.
fn main() {
    for row in lb_ternary_gpu::list_adapters() {
        let facts = (!row.bus.is_empty()).then(|| lb_ternary_gpu::card_facts::read_card_facts(&row.bus)).flatten();
        let idx = row.discrete_index.map(|i| format!("discrete:{i}")).unwrap_or_else(|| row.device_type.clone());
        match facts {
            Some(f) => println!("{idx} {}: {}", row.name, f.summary()),
            None => println!("{idx} {}: no sysfs facts (bus {:?})", row.name, row.bus),
        }
    }
}
