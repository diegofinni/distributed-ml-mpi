#include "worker_node.hpp"
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

#include <mpi.h>
#include "decentralized.hpp"
#include "lr.hpp"
#include <string>
#include <chrono>

// Timing 
using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::duration;
using chrono::milliseconds;
int num_epoch;

void update_params(vector<double>& params, int iter) {
    MPI_Send(&params.front(), params.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    if (iter < num_epoch - 1) MPI_Recv(&params.front(), params.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Nang, replace this code with lr stuff
// Outer for loop runs num_epoch times, call update_params after each iteration
// Rank variable is unecessary
void work(vector<double>& params, int num_epoch, int rank, int num_workers, string infile) {
    num_epoch = num_epoch;
    
    // Data is stored as a 2d vector. Each row is a pair of data, label.
    vector<vector<double> > data = input_data(infile);

    // partition data
    int num_rows = data.size();
    int X = rank * (num_rows / num_workers);
    int Y = X + (num_rows / num_workers);
    if (rank == num_workers) Y = data.size();
    auto start = data.begin() + X;
    auto end = data.begin() + Y;
  
    // To store the sliced vector
    vector<vector<double> > data_shard(Y - X);
    // Copy vector using copy function()
    copy(start, end, data_shard.begin());

    init_theta((data[0]).size() - 1);
    init_gradient((data[0]).size() - 1);
    int m = data_shard.size();

    auto t1 = high_resolution_clock::now();

    double* recv_buf = (double*) malloc(theta.size() * sizeof(double));
    for (int i = 0; i < num_epoch; i++) {
        // Placeholder line here
        train(data_shard, 0, 1);
        
        update_params(gradient, i); // This is a lil weird variable-name wise on the master node side but we want to send over the gradient
        reset_gradient();
    }
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, milli> ms_double = t2 - t1;

    cout << "Time of training: " << ms_double.count() << "ms" << endl;
}