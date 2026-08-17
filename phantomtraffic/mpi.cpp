#include "simulation.h"

#include <mpi.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace
{
    double deterministicRandom(int timeStep, int vehicleIndex)
    {
        std::uint32_t value = 42u;

        value ^= static_cast<std::uint32_t>(timeStep) * 0x9E3779B9u;
        value ^= static_cast<std::uint32_t>(vehicleIndex) * 0x85EBCA6Bu;

        value ^= value >> 16;
        value *= 0x7FEB352Du;
        value ^= value >> 15;
        value *= 0x846CA68Bu;
        value ^= value >> 16;

        return static_cast<double>(value) / 4294967296.0;
    }
}

double runMPI(const SimulationConfig& config)
{
    int worldRank = 0;
    int worldSize = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    if (config.numVehicles <= 0 || config.roadLength <= 0)
    {
        if (worldRank == 0)
        {
            std::cerr << "Invalid simulation configuration.\n";
        }

        return 0.0;
    }

    const int activeSize = std::min(worldSize, config.numVehicles);
    const bool isActive = worldRank < activeSize;

    MPI_Comm activeComm = MPI_COMM_NULL;

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
        MPI_Comm_rank(activeComm, &activeRank);
        MPI_Comm_size(activeComm, &processCount);

        const int baseCount = config.numVehicles / processCount;
        const int remainder = config.numVehicles % processCount;

        localVehicleCount =
            baseCount + (activeRank < remainder ? 1 : 0);

        globalStartIndex =
            activeRank * baseCount +
            std::min(activeRank, remainder);

        const int spacing =
            config.roadLength / config.numVehicles;

        current.resize(localVehicleCount);

        for (int i = 0; i < localVehicleCount; ++i)
        {
            const int globalIndex = globalStartIndex + i;

            current[i].position = globalIndex * spacing;
            current[i].velocity = 0;
        }

        next = current;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    if (isActive)
    {
        const int previousRank =
            (activeRank - 1 + processCount) % processCount;

        const int followingRank =
            (activeRank + 1) % processCount;

        for (int t = 0; t < config.timeSteps; ++t)
        {
            const int firstPosition = current.front().position;
            int followingFirstPosition = 0;

            MPI_Sendrecv(
                &firstPosition,
                1,
                MPI_INT,
                previousRank,
                0,

                &followingFirstPosition,
                1,
                MPI_INT,
                followingRank,
                0,

                activeComm,
                MPI_STATUS_IGNORE
            );

            for (int i = 0; i < localVehicleCount; ++i)
            {
                const int globalIndex = globalStartIndex + i;
                int velocity = current[i].velocity;

                // 1. Acceleration
                velocity =
                    std::min(velocity + 1, config.maxSpeed);

                // 2. Braking
                int nextPosition;

                if (i + 1 < localVehicleCount)
                {
                    // process¡£
                    nextPosition = current[i + 1].position;
                }
                else
                {
                    // followingRank¡£
                    nextPosition = followingFirstPosition;
                }

                int gap =
                    nextPosition - current[i].position - 1;

                if (gap < 0)
                {
                    gap += config.roadLength;
                }

                velocity = std::min(velocity, gap);

                // 3. Random slowdown
                if (velocity > 0 &&
                    deterministicRandom(t, globalIndex) <
                    config.slowProbability)
                {
                    --velocity;
                }

                // 4. Movement
                next[i].velocity = velocity;
                next[i].position =
                    (current[i].position + velocity) %
                    config.roadLength;
            }

            current.swap(next);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double localElapsed = MPI_Wtime() - startTime;

    double maximumElapsed = 0.0;

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
        MPI_Comm_free(&activeComm);
    }

    return maximumElapsed;
}