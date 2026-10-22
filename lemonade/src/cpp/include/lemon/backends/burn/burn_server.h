#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/wrapped_server.h"

namespace lemon { namespace backends {

class BurnServer : public WrappedServer {
public:
    // No auto-install: lemonade-cake-bridge and lemonade-burn-server must be on PATH.
    static InstallParams get_install_params(const std::string& backend,
                                            const std::string& version);

    BurnServer(const std::string& log_level,
               ModelManager* model_manager,
               BackendManager* backend_manager);
    ~BurnServer() override;

    void load(const std::string& model_name,
              const ModelInfo& model_info,
              const RecipeOptions& options,
              bool do_not_upgrade = false) override;

    void unload() override;

    json chat_completion(const json& request) override;
    json completion(const json& request) override;

    std::map<std::string, nlohmann::json> get_additional_telemetry() override;
    std::string get_additional_telemetry_url() const override;

private:
    int private_port_ = 18090;
};

namespace burn {
std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
constexpr uint32_t capabilities() { return capability_mask_of<BurnServer>(); }
}  // namespace burn

}}  // namespace lemon::backends
