#include "unpu_runtime.h"

#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>  // You need to vendor this header or add it as a dep.

namespace unpu {

using json = nlohmann::json;

bool load_program(const std::string& program_dir, Program& out_program) {
    const std::string program_path = program_dir + "/program.json";

    std::ifstream f(program_path);
    if (!f) {
        std::cerr << "Failed to open program.json at " << program_path << "\n";
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse program.json: " << e.what() << "\n";
        return false;
    }

    out_program.program_dir = program_dir;
    out_program.artifacts.clear();

    if (!j.contains("backend_artifacts")) {
        std::cerr << "program.json has no backend_artifacts array\n";
        return false;
    }

    for (const auto& a : j["backend_artifacts"]) {
        BackendArtifact art;
        art.backend = a.value("backend", "");
        art.artifact_type = a.value("artifact_type", "");
        art.path = a.value("path", "");
        if (!art.backend.empty() && !art.path.empty()) {
            out_program.artifacts.push_back(art);
        }
    }

    if (out_program.artifacts.empty()) {
        std::cerr << "program.json contains no valid backend artifacts\n";
        return false;
    }

    return true;
}

bool run(const Program& program) {
    // Phase-2 stub: just print info about the ai8x backend artifact
    std::cout << "unpu_runtime: running program in directory " << program.program_dir << "\n";
    for (const auto& a : program.artifacts) {
        std::cout << "  backend=" << a.backend
                  << " type=" << a.artifact_type
                  << " path=" << a.path << "\n";
        if (a.backend == "ai8x") {
            // Here you would call into ai8x runtime library, passing the compiled binary.
            std::cout << "  (ai8x backend stub: would invoke ai8x runtime here)\n";
        }
    }
    return true;
}

} // namespace unpu
