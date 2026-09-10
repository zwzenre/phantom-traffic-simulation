#include "simulation.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {
constexpr int LOOP_COUNT = 4;
constexpr int JUNCTION_COUNT = 4;
constexpr int JUNCTION_BATCH_SIZE = 5;

const double JUNCTION_POSITIONS[4][4] = {
    {0.875, 0.625, 0.000, 0.625},
    {0.625, 0.375, 0.000, 0.000},
    {0.000, 0.125, 0.375, 0.000},
    {0.125, 0.000, 0.125, 0.875}
};

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

bool shouldSlowDown(int timeStep, int vehicleId, double probability) {
    if (probability <= 0.0) return false;
    if (probability >= 1.0) return true;
    const double r = static_cast<double>(deterministicRandomBits(timeStep, vehicleId)) / 4294967296.0;
    return r < probability;
}

int normalizedPosition(double value, int roadLength) {
    int p = static_cast<int>(std::llround(value * roadLength));
    p %= roadLength;
    if (p < 0) p += roadLength;
    return p;
}

int loopJunctionPosition(int loopId, int junctionIndex, int roadLength) {
    return normalizedPosition(JUNCTION_POSITIONS[loopId][junctionIndex], roadLength);
}

int forwardDistance(int from, int to, int roadLength) {
    return (to - from + roadLength) % roadLength;
}

int circularDistance(int a, int b, int roadLength) {
    const int d = std::abs(a - b);
    return std::min(d, roadLength - d);
}
}

std::vector<JunctionState> createJunctionModel(const SimulationConfig& config) {
    return {
        {0, 1, loopJunctionPosition(0, 0, config.roadLength), loopJunctionPosition(1, 0, config.roadLength), -1, 0, 0},
        {1, 2, loopJunctionPosition(1, 1, config.roadLength), loopJunctionPosition(2, 1, config.roadLength), -1, 1, 0},
        {2, 3, loopJunctionPosition(2, 2, config.roadLength), loopJunctionPosition(3, 2, config.roadLength), -1, 2, 0},
        {3, 0, loopJunctionPosition(3, 3, config.roadLength), loopJunctionPosition(0, 3, config.roadLength), -1, 3, 0}
    };
}

int junctionClearanceCells(const SimulationConfig& config) {
    return std::max(1, static_cast<int>(std::ceil(config.roadLength * 0.05)));
}

void initializeVehicles(std::vector<Vehicle>& vehicles, const SimulationConfig& config) {
    vehicles.clear();
    vehicles.reserve(static_cast<std::size_t>(config.numVehicles));

    const auto junctions = createJunctionModel(config);
    const int clearance = junctionClearanceCells(config);
    const int baseCount = config.numVehicles / LOOP_COUNT;
    const int remainder = config.numVehicles % LOOP_COUNT;
    int vehicleId = 0;

    for (int loop = 0; loop < LOOP_COUNT; ++loop) {
        const int count = baseCount + (loop < remainder ? 1 : 0);
        std::vector<int> forbidden;
        for (const auto& j : junctions) {
            if (j.loopA == loop) forbidden.push_back(j.positionA);
            if (j.loopB == loop) forbidden.push_back(j.positionB);
        }

        for (int i = 0; i < count; ++i) {
            int position = static_cast<int>((static_cast<long long>(i) * config.roadLength) / std::max(1, count));
            position = (position + loop * 3 + 1) % config.roadLength;

            for (int attempt = 0; attempt < config.roadLength; ++attempt) {
                bool valid = true;
                for (int junctionPosition : forbidden) {
                    if (circularDistance(position, junctionPosition, config.roadLength) <= clearance) {
                        valid = false;
                        break;
                    }
                }
                if (valid) break;
                position = (position + 1) % config.roadLength;
            }

            vehicles.push_back({position, 0, loop, vehicleId++});
        }
    }
}

int computeGap(const std::vector<Vehicle>& vehicles, int index, const SimulationConfig& config) {
    if (index < 0 || index >= static_cast<int>(vehicles.size())) return 0;

    const Vehicle& current = vehicles[index];
    int gap = config.roadLength - 1;
    bool found = false;

    for (std::size_t j = 0; j < vehicles.size(); ++j) {
        if (static_cast<int>(j) == index) continue;
        const Vehicle& other = vehicles[j];
        if (other.loopId != current.loopId) continue;

        const int distance = forwardDistance(current.position, other.position, config.roadLength);
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

void calculateProposedVelocities(const std::vector<Vehicle>& current, std::vector<Vehicle>& next, const SimulationConfig& config, int timeStep, unsigned int randomSeed) {
    (void)randomSeed;
    next = current;

    for (std::size_t i = 0; i < current.size(); ++i) {
        int velocity = std::min(current[i].velocity + 1, config.maxSpeed);
        velocity = std::min(velocity, computeGap(current, static_cast<int>(i), config));
        if (velocity > 0 && shouldSlowDown(timeStep, current[i].id, config.slowProbability)) --velocity;
        next[i].velocity = velocity;
    }
}

int findCrossingCandidate(const std::vector<Vehicle>& vehicles, int loopId, int junctionPosition, int proposedVelocity, const SimulationConfig& config) {
    const int clearance = junctionClearanceCells(config);
    int bestIndex = -1;
    int bestDistance = std::numeric_limits<int>::max();

    for (std::size_t i = 0; i < vehicles.size(); ++i) {
        const Vehicle& v = vehicles[i];
        if (v.loopId != loopId) continue;
        const int distance = forwardDistance(v.position, junctionPosition, config.roadLength);
        if (distance > clearance && distance <= proposedVelocity + clearance && distance < bestDistance) {
            bestDistance = distance;
            bestIndex = static_cast<int>(i);
        }
    }
    return bestIndex;
}

int applyJunctionRules(const std::vector<Vehicle>& current, std::vector<Vehicle>& next, std::vector<JunctionState>& junctions, const SimulationConfig& config) {
    const int clearance = junctionClearanceCells(config);
    int waitingVehicles = 0;

    for (auto& junction : junctions) {
        bool occupiedA = false;
        bool occupiedB = false;

        for (const auto& v : current) {
            if (v.loopId != junction.loopA && v.loopId != junction.loopB) continue;
            const int jp = v.loopId == junction.loopA ? junction.positionA : junction.positionB;
            if (circularDistance(v.position, jp, config.roadLength) <= clearance) {
                if (v.loopId == junction.loopA) occupiedA = true;
                else occupiedB = true;
            }
        }

        if (junction.owner == junction.loopA && !occupiedA) {
            ++junction.batchCount;
            if (junction.batchCount >= JUNCTION_BATCH_SIZE) {
                junction.turn = junction.loopB;
                junction.batchCount = 0;
            }
            junction.owner = -1;
        } else if (junction.owner == junction.loopB && !occupiedB) {
            ++junction.batchCount;
            if (junction.batchCount >= JUNCTION_BATCH_SIZE) {
                junction.turn = junction.loopA;
                junction.batchCount = 0;
            }
            junction.owner = -1;
        }

        const int candidateA = findCrossingCandidate(current, junction.loopA, junction.positionA, config.maxSpeed, config);
        const int candidateB = findCrossingCandidate(current, junction.loopB, junction.positionB, config.maxSpeed, config);

        if (junction.owner == -1) {
            if (candidateA != -1 && candidateB != -1) {
                junction.owner = (junction.turn == junction.loopA) ? junction.loopA : junction.loopB;
            } else if (candidateA != -1) {
                junction.owner = junction.loopA;
            } else if (candidateB != -1) {
                junction.owner = junction.loopB;
            }
        }

        if (junction.owner == -1) continue;

        for (std::size_t i = 0; i < current.size(); ++i) {
            const Vehicle& v = current[i];
            if (v.loopId != junction.loopA && v.loopId != junction.loopB) continue;
            if (v.loopId == junction.owner) continue;

            const int jp = v.loopId == junction.loopA ? junction.positionA : junction.positionB;
            const int distance = forwardDistance(v.position, jp, config.roadLength);
            if (distance > clearance && distance <= next[i].velocity + clearance) {
                const int permitted = std::max(0, distance - clearance - 1);
                if (next[i].velocity > permitted) {
                    next[i].velocity = permitted;
                    ++waitingVehicles;
                }
            }
        }
    }

    return waitingVehicles;
}

void commitVehicles(std::vector<Vehicle>& current, std::vector<Vehicle>& next, const SimulationConfig& config) {
    for (auto& v : next) {
        v.position = (v.position + v.velocity) % config.roadLength;
    }
    current.swap(next);
}

void printRoad(const std::vector<Vehicle>& vehicles, const SimulationConfig& config) {
    for (int loop = 0; loop < LOOP_COUNT; ++loop) {
        std::string road(static_cast<std::size_t>(config.roadLength), '.');
        for (const auto& v : vehicles) {
            if (v.loopId == loop && v.position >= 0 && v.position < config.roadLength) {
                road[static_cast<std::size_t>(v.position)] = 'X';
            }
        }
        std::cout << "Loop " << (loop + 1) << ": " << road << '\n';
    }
}
