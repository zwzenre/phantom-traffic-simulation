#include "simulation.h"
#include "cuda.cuh"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <mpi.h>
#include <omp.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int mpiRank = 0;
    int mpiProcesses = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiProcesses);

    SimulationConfig config{};
    config.roadLength = 10000;
    config.numVehicles = 2000;
    config.maxSpeed = 5;
    config.slowProbability = 0.2;
    config.timeSteps = 5000;

    if (argc == 4) {
        try {
            config.roadLength = std::stoi(argv[1]);
            config.numVehicles = std::stoi(argv[2]);
            config.maxSpeed = std::stoi(argv[3]);
        } catch (const std::exception&) {
            if (mpiRank == 0) std::cerr << "ERROR: Parameters must be integers.\n";
            MPI_Finalize();
            return 1;
        }
    } else if (argc != 1) {
        if (mpiRank == 0) {
            std::cerr << "Usage: phantomtraffic [roadLength numVehicles maxSpeed]\n";
        }
        MPI_Finalize();
        return 1;
    }

    // There are four independent circular loops, so vehicles are distributed
    // across four roads rather than sharing one road-length capacity.
    if (config.roadLength <= 0 ||
        config.numVehicles <= 0 ||
        config.numVehicles > config.roadLength * 4 ||
        config.maxSpeed <= 0 ||
        config.timeSteps <= 0 ||
        config.slowProbability < 0.0 ||
        config.slowProbability > 1.0) {
        if (mpiRank == 0) {
            std::cerr << "ERROR: Invalid configuration. Require roadLength > 0, "
                      << "0 < vehicleCount <= 4 * roadLength, maxSpeed > 0, "
                      << "timeSteps > 0, and slowProbability in [0,1].\n";
        }
        MPI_Finalize();
        return 1;
    }

    double serialTime = 0.0;
    double openmpTime = 0.0;
    double cudaTime = 0.0;

    if (mpiRank == 0) {
        std::cout << "Threads: " << omp_get_max_threads() << '\n';
        std::cout << "MPI Processes: " << mpiProcesses << '\n';
        std::cout << "Traffic Model: 4 loops / 4 shared junctions\n";

        serialTime = runSerial(config);
        openmpTime = runOpenMP(config);
        cudaTime = runCUDA(config);
    }

    const double mpiTime = runMPI(config);

    if (mpiRank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "RESULT Serial " << serialTime * 1000.0 << "\n";
        std::cout << "RESULT OpenMP " << openmpTime * 1000.0 << "\n";
        std::cout << "RESULT CUDA " << cudaTime * 1000.0 << "\n";
        std::cout << "RESULT MPI " << mpiTime * 1000.0 << "\n";
    }

    MPI_Finalize();
    return 0;
}
