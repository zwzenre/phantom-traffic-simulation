#pragma once

#include <vector>

enum class LoopId {
    LOOP_1 = 0,
    LOOP_2 = 1,
    LOOP_3 = 2,
    LOOP_4 = 3
};

struct Vehicle {
    int position;
    int velocity;
    int loopId;
    int id;
};

struct JunctionState {
    int loopA;
    int loopB;
    int positionA;
    int positionB;
    int owner;
    int turn;
    int batchCount;
};

struct SimulationConfig {
    int roadLength;
    int numVehicles;
    int maxSpeed;
    double slowProbability;
    int timeSteps;
};

std::vector<JunctionState> createJunctionModel(const SimulationConfig& config);
void initializeVehicles(std::vector<Vehicle>& vehicles, const SimulationConfig& config);
int computeGap(const std::vector<Vehicle>& vehicles, int index, const SimulationConfig& config);
void calculateProposedVelocities(const std::vector<Vehicle>& current, std::vector<Vehicle>& next, const SimulationConfig& config, int timeStep, unsigned int randomSeed = 42);
int junctionClearanceCells(const SimulationConfig& config);
int findCrossingCandidate(const std::vector<Vehicle>& vehicles, int loopId, int junctionPosition, int proposedVelocity, const SimulationConfig& config);
int applyJunctionRules(const std::vector<Vehicle>& current, std::vector<Vehicle>& next, std::vector<JunctionState>& junctions, const SimulationConfig& config);
void commitVehicles(std::vector<Vehicle>& current, std::vector<Vehicle>& next, const SimulationConfig& config);
void printRoad(const std::vector<Vehicle>& vehicles, const SimulationConfig& config);

double runSerial(const SimulationConfig& config);
double runOpenMP(const SimulationConfig& config);
double runMPI(const SimulationConfig& config);
double runCUDA(const SimulationConfig& config);
