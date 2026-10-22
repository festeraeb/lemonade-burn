#pragma once

#include "lemon/backends/backend_descriptor.h"

namespace lemon { namespace backends { namespace burn {

// lemonade-burn: Burn-based Rust runtime with native ternary (BitNet b1.58) kernels.
// Launched through the generic Rust bridge (lemonade-cake-bridge with
// --runtime-name lemonade-burn), which spawns `lemonade-burn-server` from a
// command template and verifies its identity via /v1/models owned_by.
inline const BackendDescriptor descriptor = {
    /*recipe*/              "lemonade-burn",
    /*display_name*/        "lemonade-burn (Burn + ternary CPU kernels)",
    /*binary*/              "lemonade-cake-bridge",
    /*config_section*/      "burn",
    /*default_device*/      DEVICE_CPU,
    /*slot_policy*/         SlotPolicy::Standard,
    /*selectable_backend*/  false,
    /*uses_ctx_size*/       false,
    /*dynamic_models*/      false,
    /*options*/ {
        {"burn_args", "--burn-args", "", "ARGS",
         "lemonade-burn-server command template ({model} {port} {device} placeholders)", "lemonade-burn Options"},
        {"burn_port", "--burn-port", "", "PORT",
         "Private lemonade-burn-server port behind the Rust bridge (default 18090)", "lemonade-burn Options"},
    },
    /*support*/ {
        {"system", {"linux"}, {{"cpu", {"x86_64"}}}, "x86_64 CPU (AVX2 / AVX-512 ternary kernels)"},
    },
    /*supported_modes*/ {"chat"},
    /*required_checkpoints*/ {"main"},
    /*default_capabilities*/ {},
    /*experimental*/        true,
    /*web_display_name*/    "",
    /*rocm_channels*/       {},
    /*exposes_prometheus_metrics*/ false,
    /*rocm_requires_cwsr_fix*/ false,
    /*version_policy*/      VersionPolicy::AtLeast,
    /*self_manages_downloads*/ true,   // checkpoint is a local converted model dir
    /*takes_args*/          false,
    /*arg_variants*/        {},
    /*bin_variants*/        {},
    /*config_extra*/        nlohmann::json::object(),
};

}}}  // namespace lemon::backends::burn
