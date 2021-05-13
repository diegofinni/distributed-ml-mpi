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

    /*
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
    */

    // Initialize MPI environment
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    auto t1 = high_resolution_clock::now();

    // Data is stored as a 2d vector. Each row is a pair of data, label.
    vector<vector<double> > data = input_data(infile);

    init_theta((data[0]).size() - 1);
    init_gradient((data[0]).size() - 1);

    if (!rank) {
        init_master_node(
            theta,
            num_procs - 1,
            num_epoch,
            n_bound,
            learning_rate);
        theta = manage_workers();
    }
    else {
        // Data is stored as a 2d vector. Each row is a pair of data, label.
        vector<vector<double> > data = input_data(infile);

        int num_workers = num_procs - 1;

        // partition data
        int num_rows = data.size();
        int X = (rank-1) * (num_rows / num_workers);
        int Y = X + (num_rows / num_workers);
        if (rank == num_workers) Y = data.size();
        auto start = data.begin() + X;
        auto end = data.begin() + Y;
    
        // To store the sliced vector
        vector<vector<double> > data_shard(Y - X);
        // Copy vector using copy function()
        copy(start, end, data_shard.begin());

        work(theta,
            data_shard,
            num_epoch,
            rank,
            num_procs - 1,
            infile);
    }

    // Debugging: Print theta
    if(!rank) {
         auto t2 = high_resolution_clock::now();
        /* Print time */
        duration<double, milli> ms_double = t2 - t1;
        printf("%.3f\n", ms_double.count() / 1000);
        /*
        // Note: theta may give different values because we do not divide by m, rows of data shard, in this
        // finer-grained training model
        
        cout << "Theta: ";
        for(auto param : theta) {
            cout << param << " ";
        }
        cout << endl;
        */
    }

    MPI_Finalize();
    return 0;
}