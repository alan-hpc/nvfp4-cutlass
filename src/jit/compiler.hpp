#pragma once

// Minimal JIT: generate a .cu, drive nvcc, cache and load the cubin.
//
// Conceptually the same approach DeepGEMM takes -- write the instantiation,
// shell out to nvcc, cache by content hash, load with the driver API -- but
// implemented here so `nvfp4_gemm` builds without a DeepGEMM checkout.
//
// The cache key covers the generated source, the compiler flags and the nvcc
// version, so changing a template parameter, an include, or the toolchain all
// force a rebuild.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>

#include "../runtime/device.hpp"
#include "../runtime/exception.hpp"

namespace nvfp4_gemm::jit {

namespace fs = std::filesystem;

inline std::string get_env(const char* name, const std::string& fallback = "")
{
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : fallback;
}

/// Run a command, returning (exit code, combined output).
inline std::pair<int, std::string> run_command(const std::string& command)
{
    std::string full = command + " 2>&1";
    FILE*       pipe = popen(full.c_str(), "r");
    NVFP4_HOST_ASSERT_MSG(pipe != nullptr, "failed to spawn: " + command);

    std::string output;
    char        buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr)
        output += buffer;
    const int status = pclose(pipe);
    return {WIFEXITED(status) ? WEXITSTATUS(status) : -1, output};
}

/// A loaded kernel, owning its CUDA module.
struct Kernel
{
    CUmodule   module   = nullptr;
    CUfunction function = nullptr;

    ~Kernel()
    {
        if (module != nullptr)
            cuModuleUnload(module);
    }
};

using KernelPtr = std::shared_ptr<Kernel>;

class Compiler
{
    fs::path                                   cuda_home;
    std::string                                arch_flags;
    fs::path                                   include_root;
    fs::path                                   cache_dir;
    std::string                                flags;
    std::string                                signature;
    std::unordered_map<std::string, KernelPtr> loaded;

    /// Find an architecture flag that actually assembles the NVFP4 MMA.
    ///
    /// The tcgen05 NVFP4 instructions exist only on architecture-specific ("a")
    /// or architecture-family ("f") targets. `--gpu-architecture=sm_100f` alone
    /// is not enough: in whole-compilation mode it also embeds forward-compatible
    /// PTX, and that PTX carries `.target sm_100`, which ptxas then honours --
    /// rejecting every suffixed instruction. Naming the virtual architecture
    /// explicitly keeps the suffix on both stages.
    ///
    /// The probe compiles with `-c`, the stricter of the two modes, so the flag
    /// it returns also works for the `-cubin` compilation this JIT actually uses.
    std::string resolve_arch_flags()
    {
        if (const auto override_flags = get_env("NVFP4_ARCH_FLAGS"); not override_flags.empty())
            return override_flags;

        const auto probe_path = cache_dir / "arch_probe.cu";
        {
            std::ofstream out(probe_path);
            out << "__global__ void probe(unsigned long long a, unsigned long long b, unsigned c, unsigned d)\n"
                << "{\n"
                << "    asm volatile(\n"
                << "        \"{\\n\\t\"\n"
                << "        \".reg .pred p;\\n\\t\"\n"
                << "        \"setp.ne.b32 p, %3, 0;\\n\\t\"\n"
                << "        \"tcgen05.fence::before_thread_sync;\\n\\t\"\n"
                << "        \"tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
                   "[%2], %0, %1, %3, [%2], [%2], p;\\n\\t\"\n"
                << "        \"}\\n\" ::\"l\"(a), \"l\"(b), \"r\"(c), \"r\"(d));\n"
                << "}\n";
        }

        // Explicit virtual+real pairs first, family before architecture-specific.
        const std::vector<std::string> candidates = {
            "-gencode=arch=compute_100f,code=sm_100f",
            "-gencode=arch=compute_103f,code=sm_103f",
            "-gencode=arch=compute_103a,code=sm_103a",
            "-gencode=arch=compute_100a,code=sm_100a",
            "--gpu-architecture=sm_100f",
            "--gpu-architecture=sm_103a",
            "--gpu-architecture=sm_100a",
        };
        for (const auto& candidate : candidates)
        {
            const auto command = (cuda_home / "bin" / "nvcc").string() + " " + candidate + " -c -o " +
                                 (cache_dir / "arch_probe.o").string() + " " + probe_path.string();
            if (run_command(command).first == 0)
            {
                if (not get_env("NVFP4_JIT_DEBUG").empty())
                    std::printf("nvfp4_gemm JIT: architecture flags '%s'\n", candidate.c_str());
                return candidate;
            }
        }
        NVFP4_HOST_ASSERT_MSG(false,
                              "no architecture flag assembles tcgen05.mma kind::mxf4nvf4.block16; "
                              "needs CUDA >= 12.9 on Blackwell, or set NVFP4_ARCH_FLAGS");
        return {};
    }

public:
    /// `include_root` is the single directory the generated code is compiled
    /// against; it must contain `nvfp4_gemm/`, `cutlass/` and `cute/`.
    void init(const std::string& include_root_path, const std::string& cuda_home_path)
    {
        include_root = include_root_path;
        cuda_home    = cuda_home_path;

        cache_dir = get_env("NVFP4_JIT_CACHE_DIR");
        if (cache_dir.empty())
            cache_dir = fs::path(get_env("HOME", "/tmp")) / ".nvfp4_gemm";
        fs::create_directories(cache_dir);

        const auto [rc, version_text] = run_command((cuda_home / "bin" / "nvcc").string() + " --version");
        NVFP4_HOST_ASSERT_MSG(rc == 0, "nvcc not runnable at " + (cuda_home / "bin" / "nvcc").string());

        int major = 0, minor = 0;
        if (const auto pos = version_text.find("release "); pos != std::string::npos)
            std::sscanf(version_text.c_str() + pos + 8, "%d.%d", &major, &minor);
        // `kind::mxf4nvf4.block16` and the architecture-family targets both
        // arrived in 12.9; older nvcc can only emit sm_100a, which will not load
        // on a B300.
        NVFP4_HOST_ASSERT_MSG(major > 12 or (major == 12 and minor >= 9),
                              "CUDA >= 12.9 required, found " + std::to_string(major) + "." + std::to_string(minor));
        signature = "nvcc" + std::to_string(major) + "." + std::to_string(minor);

        arch_flags = resolve_arch_flags();

        std::ostringstream oss;
        oss << " -std=c++20 -O3 " << arch_flags << " -cubin"
            << " --expt-relaxed-constexpr --expt-extended-lambda"
            << " --diag-suppress=39,161,174,177,186,550,940"
            << " -I" << include_root.string() << " -I" << (cuda_home / "include").string();
        if (not get_env("NVFP4_JIT_DEBUG").empty())
            oss << " --ptxas-options=--verbose";
        flags = oss.str();
    }

    bool is_initialized() const
    {
        return not include_root.empty();
    }

    /// Compile `code` (or reuse a cached cubin) and return the single kernel in it.
    KernelPtr build(const std::string& name, const std::string& code)
    {
        NVFP4_HOST_ASSERT_MSG(is_initialized(), "compiler not initialized; call nvfp4_gemm._C.init()");

        const auto key = std::to_string(std::hash<std::string>{}(signature + flags + code));
        if (const auto it = loaded.find(key); it != loaded.end())
            return it->second;

        const auto dir        = cache_dir / (name + "." + key);
        const auto cubin_path = dir / "kernel.cubin";
        if (not fs::exists(cubin_path))
        {
            fs::create_directories(dir);
            const auto source_path = dir / "kernel.cu";
            // Write to a temporary and rename, so a concurrent process never
            // observes a half-written source or cubin.
            const auto tmp_source = source_path.string() + ".tmp" + std::to_string(getpid());
            {
                std::ofstream out(tmp_source);
                out << code;
            }
            fs::rename(tmp_source, source_path);

            const auto tmp_cubin = cubin_path.string() + ".tmp" + std::to_string(getpid());
            const auto command   = (cuda_home / "bin" / "nvcc").string() + " " + source_path.string() + " -o " + tmp_cubin + flags;
            if (not get_env("NVFP4_JIT_DEBUG").empty())
                std::printf("nvfp4_gemm JIT: %s\n", command.c_str());

            const auto [rc, output] = run_command(command);
            NVFP4_HOST_ASSERT_MSG(rc == 0, "nvcc failed for " + name + ":\n" + output + "\ncommand: " + command);
            fs::rename(tmp_cubin, cubin_path);
        }

        auto kernel = std::make_shared<Kernel>();
        NVFP4_CU_CHECK(cuModuleLoad(&kernel->module, cubin_path.string().c_str()));

        // The generated translation unit instantiates exactly one kernel, so
        // enumerating is unnecessary: look it up by its mangled prefix.
        NVFP4_CU_CHECK(cuModuleGetFunction(&kernel->function, kernel->module, find_symbol(cubin_path).c_str()));

        loaded[key] = kernel;
        return kernel;
    }

private:
    /// Recover the mangled name of the single kernel in a cubin.
    std::string find_symbol(const fs::path& cubin_path) const
    {
        const auto [rc, output] =
            run_command((cuda_home / "bin" / "cuobjdump").string() + " -symbols " + cubin_path.string());
        NVFP4_HOST_ASSERT_MSG(rc == 0, "cuobjdump failed on " + cubin_path.string());

        std::istringstream iss(output);
        std::string        line;
        while (std::getline(iss, line))
        {
            const auto pos = line.find("STT_FUNC");
            if (pos == std::string::npos)
                continue;
            std::istringstream ls(line);
            std::string        token, last;
            while (ls >> token)
                last = token;
            // Skip helper symbols nvcc emits alongside the kernel.
            if (last.rfind("_Z", 0) == 0 and last.find("instantiate") == std::string::npos)
                return last;
        }
        NVFP4_HOST_ASSERT_MSG(false, "no kernel symbol found in " + cubin_path.string());
        return {};
    }
};

inline Compiler* compiler = new Compiler();

}   // namespace nvfp4_gemm::jit
