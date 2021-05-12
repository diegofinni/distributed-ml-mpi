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
 
#define error_exit(fmt, ...) do { fprintf(stderr, "%s error: " fmt, __func__, ##__VA_ARGS__); exit(1); } while(0);

int proc_rank, num_procs;

int main(int argc, char* argv[]) {

    // Grab command line arguments
    int N = atoi(argv[1]);
    int mode = atoi(argv[2]); // 0 = decentralized, 1 = centralized
    
    
    // Check if arguments are valid
    if (N < 1 && mode == 0) {
        error_exit("Too few nodes for decentralized mode (At least 1 needed\n");
    }
    else if (N < 2 && mode == 1) {
        error_exit("Too few nodes for centralized mode (At least 2 needed\n");
    }
    
    // Input data
    string infile = string(argv[3]);
    string outfile = string(argv[4]);
    cout << "The input file is: " << infile << endl;
    cout << "The output file is: " << outfile << endl;
    label1 = string(argv[5]);
    label2 = string(argv[6]);

    cout << "label1 is: " << "\'" << label1 << "\'" << endl;
    cout << "label2 is: " << "\'" << label2 << "\'" << endl;

    int num_epoch = stoi(argv[7]);
    double learning_rate = stod(argv[8]);
    // Initializations
    if(!parse_flags(argc, argv, 9)) return 0;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    
    // If there are less parameters than processes, we cannot proceed
    if (num_procs > N && mode == 0) {
        MPI_Finalize();
        error_exit("Less parameters than there are processes (not allowed in decentralized mode)\n");
    }

    // Data is stored as a 2d vector. Each row is a pair of data, label.
    vector<vector<double> > data = input_data(infile);
    
    // partition data
    int num_rows = data.size();
    int X = proc_rank * (num_rows / num_procs);
    int Y = X + (num_rows / num_procs);
    if (proc_rank == num_procs - 1) Y = data.size();
    auto start = data.begin() + X;
    auto end = data.begin() + Y;
  
    // To store the sliced vector
    vector<vector<double> > data_shard(Y - X);
    // Copy vector using copy function()
    copy(start, end, data_shard.begin());
    
    /*
    // Debug: print the data shard vector
    for(auto row : data_shard) {
        for(auto cell : row) {
            printf("(%d) %.3f ", proc_rank, cell);
        }
        cout << endl;
    }
    */
    
    init_theta((data[0]).size() - 1);
    init_gradient((data[0]).size() - 1);
    init_mpi_env(proc_rank, num_procs);
    ReduceFunction f = sum_reduce;

    
    auto t1 = high_resolution_clock::now();

    int m = data_shard.size();
    for(int i=0; i<num_epoch; i++)
    {
        train(data_shard, learning_rate, 1);

        //MPI_Barrier(MPI_COMM_WORLD); 

        // Now gradient is updated based on training data.
        // Send to other nodes
        // Init MPI comm data structures and set reduce function
        reduce(gradient, gradient.size(), f);

        // update theta
        for(int j=0; j<theta.size(); j++) {
            theta[j] -= learning_rate * gradient[j] / m;
        }
        reset_gradient();
    }


    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, milli> ms_double = t2 - t1;

    cout << "Time of training: " << ms_double.count() << "ms" << endl;

    // Print results
    cout << "Theta after reduce" << endl;
    for (int i = 0; i < theta.size(); i++) {
        printf("(%d) %.3f ", proc_rank, theta[i]);
    }
    printf("\n");
    
    // Exit program
    //MPI_Finalize();
    
    return 0; 
}