#include "simulation.h"
#include "cuda.cuh"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include <omp.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int mpiRank = 0;
    int mpiProcesses = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiProcesses);

    SimulationConfig config;

    config.roadLength = 10000;
    config.numVehicles = 2000;
    config.maxSpeed = 5;
    config.slowProbability = 0.2;
    config.timeSteps = 5000;

    if (argc == 4)
    {
        try
        {
            config.roadLength = std::stoi(argv[1]);
            config.numVehicles = std::stoi(argv[2]);
            config.maxSpeed = std::stoi(argv[3]);
        }
        catch (const std::exception&)
        {
            if (mpiRank == 0)
            {
                std::cerr << "ERROR: Parameters must be integers.\n";
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (config.roadLength <= 0 || config.numVehicles <= 0 ||
        config.numVehicles > config.roadLength || config.maxSpeed <= 0)
    {
        if (mpiRank == 0)
        {
            std::cerr << "ERROR: Require roadLength > 0, 0 < vehicleCount <= roadLength, and maxSpeed > 0.\n";
        }
        MPI_Finalize();
        return 1;
    }

    double serialTime = 0.0;
    double openmpTime = 0.0;
    double cudaTime = 0.0;

    if (mpiRank == 0)
    {
        std::cout << "Threads: " << omp_get_max_threads() << std::endl;

        std::cout << "MPI Processes: " << mpiProcesses << std::endl;

        serialTime = runSerial(config);
        openmpTime = runOpenMP(config);
        cudaTime = runCUDA(config);
    }

	// Run MPI benchmark on all processes
    double mpiTime = runMPI(config);

    if (mpiRank == 0)
    {
        std::cout << std::fixed << std::setprecision(6);

        // Result output in milliseconds
        std::cout << "RESULT Serial " << serialTime * 1000.0 << "\n";
        std::cout << "RESULT OpenMP " << openmpTime * 1000.0 << "\n";
        std::cout << "RESULT CUDA " << cudaTime * 1000.0 << "\n";
        std::cout << "RESULT MPI " << mpiTime * 1000.0 << "\n";
	}

    MPI_Finalize();

    return 0;
}
