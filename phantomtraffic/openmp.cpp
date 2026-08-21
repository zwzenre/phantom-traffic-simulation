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

    // Keep one OpenMP team alive for the entire simulation.  Creating a new
    // team at every timestep adds unnecessary fork/join overhead.
#pragma omp parallel
    {
        // Every thread owns its random generator, so no locking is required.
        std::mt19937 rng(42 + omp_get_thread_num());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int t = 0; t < config.timeSteps; ++t) {
#pragma omp for schedule(static)
            for (int i = 0; i < config.numVehicles; ++i) {

                int v = current[i].velocity;

                // Acceleration
                v = std::min(v + 1, config.maxSpeed);

                // Braking
                // Inline gap calculation avoids a function call in the
                // hottest loop of the simulation.
                const int nextIndex =
                    (i + 1 == config.numVehicles) ? 0 : i + 1;
                int gap = current[nextIndex].position -
                    current[i].position - 1;
                if (gap < 0) {
                    gap += config.roadLength;
                }
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

            // Exactly one thread swaps the completed next-state buffer into
            // place. The implicit barrier guarantees all threads see the
            // new state before the next timestep begins.
#pragma omp single
            {
                current.swap(next);
            }
        }
    }

    double end = omp_get_wtime();

    return end - start;
}

