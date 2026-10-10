// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF

// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <cstdio>
#include <cuda_runtime.h>
#include <gflags/gflags.h>
#include <iostream>
#include <utils/utils.h>
#include <utils/interactor.h>
#include <utils/app_skeleton.h>

bool HybridCC();
void CleanupGraphs();


namespace hybrid_cc {
    struct App {
        static const char *Name() { return "cc"; }

        static const char *NameUpper() { return "CC"; }

        inline static bool success = true;
        static bool Single() {
            const bool result = HybridCC();
            success = success && result;
            return result;
        }

        static void Cleanup() { CleanupGraphs(); }
    };
}

//here is the main function of OOM GPU graph processing
int main(int argc, char **argv) {
    Skeleton<hybrid_cc::App> app;
    int exit = 0;
    try {
        exit = app(argc, argv);
        if (!hybrid_cc::App::success) exit = 1;
    } catch (const std::exception &error) {
        fprintf(stderr, "CC: %s\n", error.what());
        exit = 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return exit;
}
