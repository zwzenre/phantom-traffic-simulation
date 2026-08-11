#include "simulation.h"
#include <fstream>
#include <iomanip>
#include <omp.h>
#include <random>
#include <algorithm>

double runOpenMP(const SimulationConfig& config)
{
    std::vector<Vehicle> current;
    std::vector<Vehicle> next;

    initializeVehicles(current, config);
    next = current;

    double start = omp_get_wtime();

    for (int t = 0; t < config.timeSteps; t++) {

#pragma omp parallel
        {
            std::mt19937 rng(42 + omp_get_thread_num() + t * 1000);
            std::uniform_real_distribution<double> dist(0.0, 1.0);

#pragma omp for schedule(static)
            for (int i = 0; i < config.numVehicles; i++) {

                int v = current[i].velocity;

                // Acceleration
                v = std::min(v + 1, config.maxSpeed);

                // Braking
                int gap = computeGap(current, i, config);
                v = std::min(v, gap);

                // Random slowdown
                if (v > 0 && dist(rng) < config.slowProbability) {
                    v--;
                }

                // Movement
                next[i].velocity = v;
                next[i].position =
                    (current[i].position + v) % config.roadLength;
            }
        }

        current.swap(next);
    }

    double end = omp_get_wtime();

    return end - start;
}

