#include "simulation.h"

#include <algorithm>
#include <omp.h>
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

int computeGapOpenMP(const std::vector<Vehicle>& vehicles, int index, const SimulationConfig& config) {
    const Vehicle& current = vehicles[index];
    int gap = config.roadLength - 1;
    bool found = false;

    for (int j = 0; j < static_cast<int>(vehicles.size()); ++j) {
        if (j == index || vehicles[j].loopId != current.loopId) continue;
        const int distance = (vehicles[j].position - current.position + config.roadLength) % config.roadLength;
        if (distance > 0) {
            const int candidate = distance - 1;
            if (!found || candidate < gap) {
                gap = candidate;
                found = true;
            }
        }
    }
    return std::max(0, gap);
}
}

double runOpenMP(const SimulationConfig& config)
{
    std::vector<Vehicle> current;
    std::vector<Vehicle> next;
    std::vector<JunctionState> junctions = createJunctionModel(config);

    initializeVehicles(current, config);
    next = current;

    const double start = omp_get_wtime();

    for (int t = 0; t < config.timeSteps; ++t) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(current.size()); ++i) {
            int velocity = std::min(current[i].velocity + 1, config.maxSpeed);
            velocity = std::min(velocity, computeGapOpenMP(current, i, config));

            const double randomValue = static_cast<double>(deterministicRandomBits(t, current[i].id)) / 4294967296.0;
            if (velocity > 0 && randomValue < config.slowProbability) --velocity;

            next[i] = current[i];
            next[i].velocity = velocity;
        }

        applyJunctionRules(current, next, junctions, config);
        commitVehicles(current, next, config);
    }

    return omp_get_wtime() - start;
}
