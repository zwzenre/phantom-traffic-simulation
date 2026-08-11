#include "simulation.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <omp.h>

int main()
{
    SimulationConfig config;

    config.roadLength = 10000;
    config.numVehicles = 2000;
    config.maxSpeed = 5;
    config.slowProbability = 0.2;
    config.timeSteps = 5000;

    std::cout << "Threads: " << omp_get_max_threads() << std::endl;

    double serialTime = runSerial(config);
    double openmpTime = runOpenMP(config);

    double speedup = serialTime / openmpTime;

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "Serial : " << serialTime << " s" << std::endl;
    std::cout << "OpenMP : " << openmpTime << " s" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // Export benchmark.csv
    // TEST EXPORT
    std::ofstream file("benchmark.txt", std::ios::trunc);

    if (!file)
    {
        std::cerr << "Cannot open benchmark.txt for writing!" << std::endl;
    }
    else
    {
        std::cerr << "Writing benchmark.txt..." << std::endl;

        file << "Serial=" << serialTime * 1000 << "\n";
        file << "OpenMP=" << openmpTime * 1000 << "\n";
        file << "CUDA=0\n";
        file << "MPI=0\n";

        file.flush();

        std::cerr << "File state after flush: " << file.good() << std::endl;

        file.close();

        std::cerr << "benchmark.txt written successfully!" << std::endl;
    }
    return 0;
}