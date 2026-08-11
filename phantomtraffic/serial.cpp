#include "simulation.h"

#include <chrono>
#include <random>
#include <algorithm>

double runSerial(const SimulationConfig& config)
{
    std::vector<Vehicle> current;
    std::vector<Vehicle> next;

    initializeVehicles(current, config);
    next = current;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < config.timeSteps; t++) {

        for (int i = 0; i < config.numVehicles; i++) {

            int v = current[i].velocity;

            // 1. Acceleration
            v = std::min(v + 1, config.maxSpeed);

            // 2. Braking
            int gap = computeGap(current, i, config);
            v = std::min(v, gap);

            // 3. Random slowdown
            if (v > 0 && dist(rng) < config.slowProbability) {
                v--;
            }

            // 4. Movement
            next[i].velocity = v;
            next[i].position = (current[i].position + v) % config.roadLength;
        }

        current.swap(next);

    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(end - start).count();
}