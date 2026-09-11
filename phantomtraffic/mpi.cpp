#include "simulation.h"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

    unsigned int deterministicRandomBits(int timeStep, int vehicleId)
    {
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

    int localStart(int rank, int worldSize, int count)
    {
        const int base = count / worldSize;
        const int remainder = count % worldSize;

        return rank * base + std::min(rank, remainder);
    }

    int localCount(int rank, int worldSize, int count)
    {
        const int base = count / worldSize;
        const int remainder = count % worldSize;

        return base + (rank < remainder ? 1 : 0);
    }

} // namespace

double runMPI(const SimulationConfig& config)
{
    int rank = 0;
    int worldSize = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    // Every rank keeps the same current state.
    std::vector<Vehicle> current;
    std::vector<Vehicle> localNext;
    std::vector<Vehicle> gatheredNext;
    std::vector<JunctionState> junctions;

    initializeVehicles(current, config);

    gatheredNext.resize(current.size());
    junctions = createJunctionModel(config);

    // Determine which vehicles this rank will calculate.
    const int start = localStart(
        rank,
        worldSize,
        config.numVehicles
    );

    const int count = localCount(
        rank,
        worldSize,
        config.numVehicles
    );

    localNext.resize(static_cast<std::size_t>(count));

    // Number of vehicles and starting index for every rank.
    std::vector<int> counts(worldSize);
    std::vector<int> displacements(worldSize);

    for (int currentRank = 0;
        currentRank < worldSize;
        ++currentRank)
    {
        counts[currentRank] = localCount(
            currentRank,
            worldSize,
            config.numVehicles
        );

        displacements[currentRank] = localStart(
            currentRank,
            worldSize,
            config.numVehicles
        );
    }

    // MPI_Allgatherv uses MPI_BYTE, so counts must be converted
    // from numbers of vehicles to numbers of bytes.
    std::vector<int> byteCounts(worldSize);
    std::vector<int> byteDisplacements(worldSize);

    for (int currentRank = 0;
        currentRank < worldSize;
        ++currentRank)
    {
        byteCounts[currentRank] =
            counts[currentRank]
            * static_cast<int>(sizeof(Vehicle));

        byteDisplacements[currentRank] =
            displacements[currentRank]
            * static_cast<int>(sizeof(Vehicle));
    }

    // Ensure all ranks begin timing together.
    MPI_Barrier(MPI_COMM_WORLD);

    const double startTime = MPI_Wtime();

    for (int timeStep = 0;
        timeStep < config.timeSteps;
        ++timeStep)
    {
        // Each rank calculates only its assigned vehicles.
        for (int localIndex = 0;
            localIndex < count;
            ++localIndex)
        {
            const int globalIndex = start + localIndex;
            const Vehicle& vehicle = current[globalIndex];

            int velocity = std::min(
                vehicle.velocity + 1,
                config.maxSpeed
            );

            velocity = std::min(
                velocity,
                computeGap(current, globalIndex, config)
            );

            const double randomValue =
                static_cast<double>(
                    deterministicRandomBits(
                        timeStep,
                        vehicle.id
                    )
                    ) / 4294967296.0;

            if (
                velocity > 0 &&
                randomValue < config.slowProbability
                )
            {
                --velocity;
            }

            localNext[localIndex] = vehicle;
            localNext[localIndex].velocity = velocity;
        }

        // Combine the proposed results calculated by every rank.
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

        /*
         * Every rank now has the same current state and the same
         * gatheredNext state.
         *
         * Junction rules are deterministic, so every rank can apply
         * the same rules locally. This removes the need for Rank 0
         * to broadcast the complete vehicle array afterward.
         */
        applyJunctionRules(
            current,
            gatheredNext,
            junctions,
            config
        );

        commitVehicles(
            current,
            gatheredNext,
            config
        );
    }

    const double localElapsed =
        MPI_Wtime() - startTime;

    double maximumElapsed = 0.0;

    // MPI runtime is determined by the slowest rank.
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