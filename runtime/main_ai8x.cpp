#include "unpu_runtime.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: unpu_ai8x_runner <program_dir>\n";
        return EXIT_FAILURE;
    }

    std::string program_dir = argv[1];

    unpu::Program program;
    if (!unpu::load_program(program_dir, program)) {
        std::cerr << "Failed to load program from " << program_dir << "\n";
        return EXIT_FAILURE;
    }

    if (!unpu::run(program)) {
        std::cerr << "Program run failed\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
