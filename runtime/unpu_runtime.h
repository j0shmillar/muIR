#pragma once

#include <string>
#include <vector>

// Very small C++ runtime interface for µPrograms targeting ai8x-style backends.
//
// In real life, this would:
//   - parse the serialized program
//   - allocate buffers
//   - call into ai8x runtime APIs
// For now, we just parse out the ai8x backend artifact and expose it.

namespace unpu {

struct BackendArtifact {
    std::string backend;       // "ai8x"
    std::string artifact_type; // "onnx" (for now)
    std::string path;          // relative to program_dir
};

struct Program {
    std::string program_dir;
    std::vector<BackendArtifact> artifacts;
};

bool load_program(const std::string& program_dir, Program& out_program);

// In a future phase, this might take tensors, etc.
bool run(const Program& program);

} // namespace unpu
