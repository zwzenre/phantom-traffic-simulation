#include "cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <vector>

namespace
{
    // Fast register-level integer hashing for pseudo-random slowdown
    __device__ __forceinline__ bool shouldSlowDown(int timeStep, int vehicleIndex, unsigned int threshold)
    {
        unsigned int value = 42u;
        value ^= static_cast<unsigned int>(timeStep) * 0x9E3779B9u;
        value ^= static_cast<unsigned int>(vehicleIndex) * 0x85EBCA6Bu;

        value ^= value >> 16;
        value *= 0x7FEB352Du;
        value ^= value >> 15;
        value *= 0x846CA68Bu;
        value ^= value >> 16;

        return value < threshold;
    }

    // High-occupancy CUDA Kernel with SoA layout
    __global__ void __launch_bounds__(256, 4) stepTrafficKernelSoA(
            const int* __restrict__ pos_curr,
            const int* __restrict__ vel_curr,
            int* __restrict__ pos_next,
            int* __restrict__ vel_next,
            int numVehicles,
            int roadLength,
            int maxSpeed,
            unsigned int slowThreshold,
            int timeStep)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numVehicles) return;

        const int pos = pos_curr[idx];
        int vel = vel_curr[idx];

        // 1. Acceleration
        vel = (vel + 1 < maxSpeed) ? (vel + 1) : maxSpeed;

        // 2. Coalesced read of the leading vehicle's position
        const int nextIdx = (idx + 1 == numVehicles) ? 0 : (idx + 1);
        const int nextPos = pos_curr[nextIdx];

        int gap = nextPos - pos - 1;
        if (gap < 0)
        {
            gap += roadLength;
        }

        // 3. Deceleration / Braking
        vel = (vel < gap) ? vel : gap;

        // 4. Random slowdown
        if (vel > 0 && shouldSlowDown(timeStep, idx, slowThreshold))
        {
            --vel;
        }

        // 5. Movement and cyclic wrapping
        int newPos = pos + vel;
        if (newPos >= roadLength)
        {
            newPos -= roadLength;
        }

        vel_next[idx] = vel;
        pos_next[idx] = newPos;
    }
}

double runCUDA(const SimulationConfig& config)
{
    if (config.numVehicles <= 0 || config.roadLength <= 0)
    {
        return 0.0;
    }

    // 1. Prepare initial host data
    std::vector<Vehicle> h_vehicles(config.numVehicles);
    initializeVehicles(h_vehicles, config);

    std::vector<int> h_pos(config.numVehicles);
    std::vector<int> h_vel(config.numVehicles);

    for (int i = 0; i < config.numVehicles; ++i)
    {
        h_pos[i] = h_vehicles[i].position;
        h_vel[i] = h_vehicles[i].velocity;
    }

    const size_t bytesInt = config.numVehicles * sizeof(int);

    // 2. Allocate Device Memory with SoA layout
    int *d_pos_curr = nullptr, *d_pos_next = nullptr;
    int *d_vel_curr = nullptr, *d_vel_next = nullptr;

    cudaMalloc(&d_pos_curr, bytesInt);
    cudaMalloc(&d_pos_next, bytesInt);
    cudaMalloc(&d_vel_curr, bytesInt);
    cudaMalloc(&d_vel_next, bytesInt);

    cudaMemcpy(d_pos_curr, h_pos.data(), bytesInt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_curr, h_vel.data(), bytesInt, cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (config.numVehicles + threadsPerBlock - 1) / threadsPerBlock;

    // Safe 32-bit threshold scaling
    const unsigned int slowThreshold = static_cast<unsigned int>(config.slowProbability * 4294967295.0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 3. Execution Loop with Ping-Pong Buffer Swapping
    for (int t = 0; t < config.timeSteps; ++t)
    {
        stepTrafficKernelSoA<<<blocksPerGrid, threadsPerBlock>>>(
                d_pos_curr,
                d_vel_curr,
                d_pos_next,
                d_vel_next,
                config.numVehicles,
                config.roadLength,
                config.maxSpeed,
                slowThreshold,
                t
        );

        std::swap(d_pos_curr, d_pos_next);
        std::swap(d_vel_curr, d_vel_next);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_pos_curr);
    cudaFree(d_pos_next);
    cudaFree(d_vel_curr);
    cudaFree(d_vel_next);

    return static_cast<double>(elapsedMs) / 1000.0;
}