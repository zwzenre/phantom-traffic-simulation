#include "simulation.h"
#include <iostream>
#include <fstream>
#include <iomanip>

#include <omp.h>
#include <mpi.h>

int main()
{
	// Initialize MPI
    MPI_Init(nullptr, nullptr);

    int mpiRank = 0;
    int mpiProcesses = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiProcesses);
    //================================================

    SimulationConfig config;

    config.roadLength = 10000;
    config.numVehicles = 2000;
    config.maxSpeed = 5;
    config.slowProbability = 0.2;
    config.timeSteps = 5000;

    //std::cout << "Threads: " << omp_get_max_threads() << std::endl;

    double serialTime = 0.0;
    double openmpTime = 0.0;

	// if (mpiRank == 0), run the serial and OpenMP benchmarks only on the root process
    if (mpiRank == 0)
    {
        std::cout << "Threads: " << omp_get_max_threads() << std::endl;

        std::cout << "MPI Processes: " << mpiProcesses << std::endl;

        serialTime = runSerial(config);
        openmpTime = runOpenMP(config);
    }

	// Run MPI benchmark on all processes
    double mpiTime = runMPI(config);

    if (mpiRank == 0)
    {
        double speedup = serialTime / openmpTime;
        double mpiSpeedup = serialTime / mpiTime;

        std::cout << std::fixed << std::setprecision(6);

        std::cout << "Serial : " << serialTime << " s" << std::endl;
        std::cout << "OpenMP : " << openmpTime << " s" << std::endl;
        std::cout << "Speedup: " << speedup << "x" << std::endl;
        std::cout << "MPI    : " << mpiTime << " s" << std::endl;
        std::cout << "MPI Speedup: " << mpiSpeedup << "x" << std::endl;

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
            file << "MPI=" << mpiTime * 1000 << "\n";

            file.flush();

            std::cerr << "File state after flush: " << file.good() << std::endl;

            file.close();

            std::cerr << "benchmark.txt written successfully!" << std::endl;
        }
	}

    MPI_Finalize();
    
    return 0;
}