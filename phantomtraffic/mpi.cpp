#include "simulation.h"

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace
{
    constexpr int boundaryTag = 0;

    // Return the same deterministic 32-bit random value as the
    // original deterministicRandom(), without converting it to double.
    inline std::uint32_t deterministicRandomBits(
        int timeStep,
        int vehicleIndex)
    {
        std::uint32_t value = 42u;

        value ^=
            static_cast<std::uint32_t>(timeStep) *
            0x9E3779B9u;

        value ^=
            static_cast<std::uint32_t>(vehicleIndex) *
            0x85EBCA6Bu;

        value ^= value >> 16;
        value *= 0x7FEB352Du;
        value ^= value >> 15;
        value *= 0x846CA68Bu;
        value ^= value >> 16;

        return value;
    }

    // Convert the probability once before the simulation.
    // This avoids floating-point division/comparison for every vehicle.
    std::uint64_t makeSlowdownThreshold(double probability)
    {
        constexpr std::uint64_t randomRange =
            std::uint64_t{ 1 } << 32;

        if (!(probability > 0.0))
        {
            return 0;
        }

        if (probability >= 1.0)
        {
            return randomRange;
        }

        return static_cast<std::uint64_t>(
            std::ceil(
                probability *
                static_cast<double>(randomRange)
            )
            );
    }

    inline void updateVehicle(
        const Vehicle& vehicle,
        int nextPosition,
        int timeStep,
        int globalIndex,
        int roadLength,
        int maxSpeed,
        std::uint64_t slowdownThreshold,
        Vehicle& updatedVehicle)
    {
        // 1. Acceleration
        int velocity =
            std::min(vehicle.velocity + 1, maxSpeed);

        // 2. Braking
        int gap =
            nextPosition - vehicle.position - 1;

        if (gap < 0)
        {
            gap += roadLength;
        }

        velocity = std::min(velocity, gap);

        // 3. Random slowdown
        if (velocity > 0 &&
            static_cast<std::uint64_t>(
                deterministicRandomBits(
                    timeStep,
                    globalIndex
                )
                ) < slowdownThreshold)
        {
            --velocity;
        }

        // 4. Movement
        int newPosition =
            vehicle.position + velocity;

        // After braking:
        // velocity <= gap <= roadLength - 1.
        // Therefore the vehicle can wrap at most once.
        if (newPosition >= roadLength)
        {
            newPosition -= roadLength;
        }

        updatedVehicle.position = newPosition;
        updatedVehicle.velocity = velocity;
    }
}

double runMPI(const SimulationConfig& config)
{
    int worldRank = 0;
    int worldSize = 1;

    MPI_Comm_rank(
        MPI_COMM_WORLD,
        &worldRank
    );

    MPI_Comm_size(
        MPI_COMM_WORLD,
        &worldSize
    );

    if (config.numVehicles <= 0 ||
        config.roadLength <= 0)
    {
        if (worldRank == 0)
        {
            std::cerr
                << "Invalid simulation configuration.\n";
        }

        return 0.0;
    }

    const int activeSize =
        std::min(
            worldSize,
            config.numVehicles
        );

    const bool isActive =
        worldRank < activeSize;

    MPI_Comm activeComm =
        MPI_COMM_NULL;

    MPI_Comm_split(
        MPI_COMM_WORLD,
        isActive ? 0 : MPI_UNDEFINED,
        worldRank,
        &activeComm
    );

    std::vector<Vehicle> current;
    std::vector<Vehicle> next;

    int activeRank = -1;
    int processCount = 0;
    int localVehicleCount = 0;
    int globalStartIndex = 0;

    if (isActive)
    {
        MPI_Comm_rank(
            activeComm,
            &activeRank
        );

        MPI_Comm_size(
            activeComm,
            &processCount
        );

        const int baseCount =
            config.numVehicles /
            processCount;

        const int remainder =
            config.numVehicles %
            processCount;

        localVehicleCount =
            baseCount +
            (activeRank < remainder ? 1 : 0);

        globalStartIndex =
            activeRank * baseCount +
            std::min(
                activeRank,
                remainder
            );

        const int spacing =
            config.roadLength /
            config.numVehicles;

        current.resize(
            localVehicleCount
        );

        // next will be completely overwritten during every step.
        // There is no need to copy current into it.
        next.resize(
            localVehicleCount
        );

        for (int i = 0;
            i < localVehicleCount;
            ++i)
        {
            const int globalIndex =
                globalStartIndex + i;

            current[i].position =
                globalIndex * spacing;

            current[i].velocity = 0;
        }
    }

    const int roadLength =
        config.roadLength;

    const int maxSpeed =
        config.maxSpeed;

    const std::uint64_t slowdownThreshold =
        makeSlowdownThreshold(
            config.slowProbability
        );

    // Synchronize only before timing so every process starts
    // the benchmark from approximately the same point.
    MPI_Barrier(
        MPI_COMM_WORLD
    );

    const double startTime =
        MPI_Wtime();

    if (isActive)
    {
        const int previousRank =
            (activeRank - 1 + processCount) %
            processCount;

        const int followingRank =
            (activeRank + 1) %
            processCount;

        for (int timeStep = 0;
            timeStep < config.timeSteps;
            ++timeStep)
        {
            // For one process, the next partition is ourselves.
            int followingFirstPosition =
                current.front().position;

            // Avoid MPI self-communication when running with
            // only one active process.
            if (processCount > 1)
            {
                const int firstPosition =
                    current.front().position;

                MPI_Sendrecv(
                    &firstPosition,
                    1,
                    MPI_INT,
                    previousRank,
                    boundaryTag,

                    &followingFirstPosition,
                    1,
                    MPI_INT,
                    followingRank,
                    boundaryTag,

                    activeComm,
                    MPI_STATUS_IGNORE
                );
            }

            // All vehicles except the final local vehicle use
            // the position of another locally stored vehicle.
            
            // Processing the boundary vehicle separately removes
            // one conditional branch from this hot loop.
            for (int i = 0;
                i + 1 < localVehicleCount;
                ++i)
            {
                updateVehicle(
                    current[i],
                    current[i + 1].position,
                    timeStep,
                    globalStartIndex + i,
                    roadLength,
                    maxSpeed,
                    slowdownThreshold,
                    next[i]
                );
            }

            // The final local vehicle depends on the first vehicle
            // stored by the following MPI process.
            const int lastLocalIndex =
                localVehicleCount - 1;

            updateVehicle(
                current[lastLocalIndex],
                followingFirstPosition,
                timeStep,
                globalStartIndex +
                lastLocalIndex,
                roadLength,
                maxSpeed,
                slowdownThreshold,
                next[lastLocalIndex]
            );

            current.swap(next);
        }
    }

    const double localElapsed =
        MPI_Wtime() - startTime;

    double maximumElapsed = 0.0;

    // Allreduce already synchronizes all processes.
    // The previous MPI_Barrier before this call was redundant.
    MPI_Allreduce(
        &localElapsed,
        &maximumElapsed,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        MPI_COMM_WORLD
    );

    if (activeComm != MPI_COMM_NULL)
    {
        MPI_Comm_free(
            &activeComm
        );
    }

    return maximumElapsed;
}