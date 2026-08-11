#pragma once

#include <vector>

struct Vehicle {
    int position;
    int velocity;
};

struct SimulationConfig {
    int roadLength;
    int numVehicles;
    int maxSpeed;
    double slowProbability;
    int timeSteps;
};

void initializeVehicles(std::vector<Vehicle>& vehicles,
    const SimulationConfig& config);

int computeGap(const std::vector<Vehicle>& vehicles,
    int index,
    const SimulationConfig& config);

void printRoad(const std::vector<Vehicle>& vehicles,
    const SimulationConfig& config);

double runSerial(const SimulationConfig& config);
double runOpenMP(const SimulationConfig& config);