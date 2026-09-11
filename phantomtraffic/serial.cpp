#include "simulation.h"

#include <algorithm>
#include <chrono>
#include <vector>

double runSerial(const SimulationConfig& config)
{
    std::vector<Vehicle> current;
    std::vector<Vehicle> next;
    std::vector<JunctionState> junctions = createJunctionModel(config);

    initializeVehicles(current, config);
    next = current;

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < config.timeSteps; ++t) {
        calculateProposedVelocities(current, next, config, t);
        applyJunctionRules(current, next, junctions, config);
        commitVehicles(current, next, config);
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}
