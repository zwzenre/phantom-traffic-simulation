#include "simulation.h"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
unsigned int deterministicRandomBits(int timeStep, int vehicleId) {
    unsigned int value = 42u;
    value ^= static_cast<unsigned int>(timeStep) * 0x9E3779B9u;
    value ^= static_cast<unsigned int>(vehicleId) * 0x85EBCA6Bu;
    value ^= value >> 16;
    value *= 0x7FEB352Du;
    value ^= value >> 15;
    value *= 0x846CA68Bu;
    value ^= value >> 16;
    return value;
}

int localStart(int rank, int worldSize, int count) {
    const int base = count / worldSize;
    const int remainder = count % worldSize;
    return rank * base + std::min(rank, remainder);
}

int localCount(int rank, int worldSize, int count) {
    const int base = count / worldSize;
    const int remainder = count % worldSize;
    return base + (rank < remainder ? 1 : 0);
}
}

double runMPI(const SimulationConfig& config)
{
    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    std::vector<Vehicle> current;
    std::vector<Vehicle> localNext;
    std::vector<Vehicle> gatheredNext;
    std::vector<JunctionState> junctions;

    initializeVehicles(current, config);
    gatheredNext.resize(current.size());
    junctions = createJunctionModel(config);

    const int start = localStart(rank, worldSize, config.numVehicles);
    const int count = localCount(rank, worldSize, config.numVehicles);

    std::vector<int> counts(worldSize);
    std::vector<int> displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = localCount(r, worldSize, config.numVehicles);
        displacements[r] = localStart(r, worldSize, config.numVehicles);
    }

    localNext.resize(static_cast<std::size_t>(count));

    std::vector<int> byteCounts(worldSize);
    std::vector<int> byteDisplacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        byteCounts[r] = counts[r] * static_cast<int>(sizeof(Vehicle));
        byteDisplacements[r] = displacements[r] * static_cast<int>(sizeof(Vehicle));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    for (int t = 0; t < config.timeSteps; ++t) {
        for (int i = 0; i < count; ++i) {
            const int globalIndex = start + i;
            const Vehicle& vehicle = current[globalIndex];

            int velocity = std::min(vehicle.velocity + 1, config.maxSpeed);
            velocity = std::min(velocity, computeGap(current, globalIndex, config));

            const double randomValue = static_cast<double>(deterministicRandomBits(t, vehicle.id)) / 4294967296.0;
            if (velocity > 0 && randomValue < config.slowProbability) --velocity;

            localNext[i] = vehicle;
            localNext[i].velocity = velocity;
        }

        MPI_Allgatherv(
            localNext.data(),
            count * static_cast<int>(sizeof(Vehicle)),
            MPI_BYTE,
            gatheredNext.data(),
            byteCounts.data(),
            byteDisplacements.data(),
            MPI_BYTE,
            MPI_COMM_WORLD
        );

        if (rank == 0) {
            applyJunctionRules(current, gatheredNext, junctions, config);
            commitVehicles(current, gatheredNext, config);
        }

        MPI_Bcast(
            current.data(),
            config.numVehicles * static_cast<int>(sizeof(Vehicle)),
            MPI_BYTE,
            0,
            MPI_COMM_WORLD
        );
    }

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

    return maximumElapsed;
}
