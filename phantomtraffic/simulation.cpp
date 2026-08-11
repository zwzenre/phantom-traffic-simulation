#include "simulation.h"

#include <iostream>
#include <algorithm>

void initializeVehicles(std::vector<Vehicle>& vehicles,
    const SimulationConfig& config)
{
    vehicles.clear();

    int spacing = config.roadLength / config.numVehicles;

    for (int i = 0; i < config.numVehicles; i++) {
        vehicles.push_back({ i * spacing, 0 });
    }
}

int computeGap(const std::vector<Vehicle>& vehicles, int index, const SimulationConfig& config)
{
    int next = (index + 1) % vehicles.size();

    int gap = vehicles[next].position - vehicles[index].position - 1;

    if (gap < 0) {
        gap += config.roadLength;
    }

    return gap;
}

void printRoad(const std::vector<Vehicle>& vehicles, const SimulationConfig& config)
{
    std::string road(config.roadLength, '.');

    for (const auto& v : vehicles) {
        road[v.position % config.roadLength] = 'X';
    }

    std::cout << road << std::endl;
}