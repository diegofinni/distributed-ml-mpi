#include <iostream>
#include <stdlib.h>
#include <string>
#include <mpi.h>
#include "master_node.hpp"
#include "worker_node.hpp"
#include "lr.hpp"

#include <chrono>

// Timing 
using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::duration;
using chrono::milliseconds;


int main(int argc, char* argv[]) {

    int num_epoch = atoi(argv[1]);
    double learning_rate = stod(argv[2]);
    int n_bound = atoi(argv[3]);
    string infile = string(argv[4]);
    string outfile = string(argv[5]);
    label1 = string(argv[6]);
    label2 = string(argv[7]);
    string intro = "***************************************\nCentralized Logistic Regression\n***************************************\n";

    // Print out program intro
    cout << intro << endl;
    cout << "Number of epochs: " << num_epoch << endl;
    cout << "Learning rate: " << learning_rate << endl;
    cout << "Staleness bound: " << n_bound << endl;
    cout << "The input file is: " << infile << endl;
    cout << "The output file is: " << outfile << endl;
    cout << "label1 is: " << "\'" << label1 << "\'" << endl;
    cout << "label2 is: " << "\'" << label2 << "\'" << endl;
    cout << "***************************************" << endl;

    // Initialize MPI environment
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    auto t1 = high_resolution_clock::now();

    vector<double> init_params(10, 0);
    if (!rank) {
        init_master_node(init_params, 
            num_procs - 1,
            num_epoch,
            n_bound,
            learning_rate);
        manage_workers();
    }
    else {
        work(init_params,
            num_epoch,
            rank,
            num_procs - 1,
            infile);
    }

    auto t2 = high_resolution_clock::now();
        /* Getting number of milliseconds as a double. */
    duration<double, milli> ms_double = t2 - t1;

    cout << "Time of training: " << ms_double.count() << "ms" << endl;

    // Print parameters here

    MPI_Finalize();
    return 0;
}