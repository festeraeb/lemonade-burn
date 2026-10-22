#include "lemon/backends/burn/burn_server.h"

#include "lemon/backends/backend_ops.h"
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backends/burn/burn.h"
#include "lemon/utils/aixlog.hpp"
#include "lemon/utils/http_client.h"
#include "lemon/utils/process_manager.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace lemon { namespace backends {

namespace {
constexpr const char* kRuntimeName = "lemonade-burn";
// `exec` so the bridge's kill reaches the server itself, not a wrapper shell.
constexpr const char* kDefaultCommand =
    "exec lemonade-burn-server --model {model} --port {port}";
}  // namespace

InstallParams BurnServer::get_install_params(const std::string& /*backend*/,
                                             const std::string& /*version*/) {
    return {};
}

BurnServer::BurnServer(const std::string& log_level,
                       ModelManager* model_manager,
                       BackendManager* backend_manager)
    : WrappedServer("lemonade-burn", log_level, model_manager, backend_manager) {}

BurnServer::~BurnServer() {
    unload();
}

void BurnServer::load(const std::string& model_name,
                      const ModelInfo& model_info,
                      const RecipeOptions& options,
                      bool /*do_not_upgrade*/) {
    const std::string checkpoint = model_info.checkpoint();
    if (checkpoint.empty()) {
        throw std::runtime_error("lemonade-burn model checkpoint is empty for: " + model_name);
    }

    std::string configured_port = options.get_option("burn_port");
    if (!configured_port.empty()) {
        private_port_ = std::stoi(configured_port);
    }

    port_ = choose_port();
    if (port_ == 0) {
        throw std::runtime_error("Failed to allocate Lemonade private port for lemonade-burn bridge");
    }

    std::string bridge_executable =
        BackendUtils::get_backend_binary_path(*burn::spec(), "system");

    std::string command = options.get_option("burn_args");
    if (command.empty()) {
        command = kDefaultCommand;
    }

    std::vector<std::string> args = {
        "--listen-port", std::to_string(port_),
        "--cake-port", std::to_string(private_port_),
        "--model", checkpoint,
        "--runtime-name", kRuntimeName,
        "--cake-command", command,
        "--log-level", log_level_,
    };

    LOG(INFO, "Burn") << "Starting bridge on port " << port_ << " for lemonade-burn model "
                      << checkpoint << std::endl;

    const bool inherit_output = (log_level_ == "info") || is_debug();
    set_process_handle(utils::ProcessManager::start_process(
                           bridge_executable, args, "", inherit_output, true, {}),
                       bridge_executable, args);

    if (!wait_for_ready("/health")) {
        unload();
        throw std::runtime_error("lemonade-burn bridge failed to become ready");
    }

    const json fingerprint = forward_get_request("/internal/runtime-fingerprint", 5);
    if (!fingerprint.value("runtime_identity_verified", false) ||
        fingerprint.value("observed_runtime", std::string{}) != kRuntimeName) {
        unload();
        throw std::runtime_error(
            "lemonade-burn runtime identity verification failed; refusing mislabeled backend");
    }

    start_backend_watchdog("/health");
    LOG(INFO, "Burn") << "lemonade-burn backend ready on bridge port " << port_ << std::endl;
}

void BurnServer::unload() {
    stop_backend_watchdog();
    const ProcessHandle handle = consume_process_handle_for_cleanup();
    if (has_process_handle(handle)) {
        utils::ProcessManager::stop_process(handle);
    }
}

json BurnServer::chat_completion(const json& request) {
    return forward_request("/v1/chat/completions", request);
}

json BurnServer::completion(const json& request) {
    return forward_request("/v1/completions", request);
}

std::map<std::string, nlohmann::json> BurnServer::get_additional_telemetry() {
    if (get_backend_port() <= 0) {
        return {};
    }
    std::map<std::string, nlohmann::json> out;
    try {
        out["burn_topology"] = forward_get_request("/api/v1/topology", 2);
    } catch (...) {
        out["burn_topology"] = nullptr;
    }
    try {
        out["runtime_fingerprint"] = forward_get_request("/internal/runtime-fingerprint", 2);
    } catch (...) {
        out["runtime_fingerprint"] = nullptr;
    }
    return out;
}

std::string BurnServer::get_additional_telemetry_url() const {
    if (get_backend_port() <= 0) {
        return "";
    }
    return get_base_url() + "/api/v1/topology";
}

}}  // namespace lemon::backends

namespace lemon { namespace backends { namespace burn {

namespace {
// Models for this backend are local directories written by `lb-convert`
// (model.lbt.safetensors + config.json + tokenizer.json); nothing is fetched
// from a hub, so "downloaded" means the converted file exists.
class BurnOps : public BackendOps {
public:
    std::string resolve_checkpoint_path(const ModelInfo& info,
                                        const CheckpointResolveContext& ctx) const override {
        (void)ctx;
        return info.checkpoint();
    }
    bool is_downloaded(const ModelInfo& info, const BackendOpsContext& ctx) const override {
        (void)ctx;
        std::error_code ec;
        return std::filesystem::is_regular_file(
            std::filesystem::path(info.checkpoint()) / "model.lbt.safetensors", ec);
    }
    void download_model(const ModelInfo& info, bool do_not_upgrade,
                        DownloadProgressCallback progress,
                        const BackendOpsContext& ctx) const override {
        (void)do_not_upgrade;
        (void)progress;
        (void)ctx;
        throw std::runtime_error("lemonade-burn models are local directories produced by lb-convert; '" +
                                 info.checkpoint() + "' has no model.lbt.safetensors");
    }
};
}  // namespace

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<BurnServer>(ctx);
}

const BackendSpec* spec() {
    return make_spec<BurnServer>(descriptor);
}

const BackendOps* ops() {
    static const BurnOps kOps;
    return &kOps;
}

}}}  // namespace lemon::backends::burn
