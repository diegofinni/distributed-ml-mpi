
#include "lr.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

// Timing 
using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::duration;
using chrono::milliseconds;

int main(int argc, char **argv){ 
    if (argc < 3) {
        cout << "Missing Input and/or Output file! Please see usage: \n";
        print_usage();
        return -1;
    }
    string infile = string(argv[1]);
    string outfile = string(argv[2]);
    cout << "The input file is: " << infile << endl;
    cout << "The output file is: " << outfile << endl;
    label1 = string(argv[3]);
    label2 = string(argv[4]);
    cout << "label1 is: " << "\'" << label1 << "\'" << endl;
    cout << "label2 is: " << "\'" << label2 << "\'" << endl;

    // Initializations
    if(!parse_flags(argc, argv, 5)) return 0;
    // Data is stored as a 2d vector. Each row is a pair of data, label.
    vector<vector<double> > data = input_data(infile);
    
    int proc_rank = 4;
    int num_procs = 5;
    // partition data
    int num_rows = data.size();
    int X = proc_rank * (num_rows / num_procs);
    int Y = X + (num_rows / num_procs);
    if(proc_rank == num_procs - 1) Y = data.size();
    auto start = data.begin() + X;
    auto end = data.begin() + Y;
  
    // To store the sliced vector
    vector<vector<double> > data_shard(Y - X + 1);
    // Copy vector using copy function()
    copy(start, end, data_shard.begin());
    
    /*
    // Debug: print the data vector
    for(auto row : data) {
        for(auto cell : row) {
            cout << cell << " ";
        }
        cout << endl;
    }
    */
    
    init_theta((data[0]).size() - 1);

    double learning_rate = 0.01;
    int num_epoch = 2000;

    auto t1 = high_resolution_clock::now();
    train(data, learning_rate, num_epoch);
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, milli> ms_double = t2 - t1;

    cout << "Time of training: " << ms_double.count() << "ms" << endl;

    // Now theta is updated based on training data.
    // Send to other nodes

    // Debugging: Print theta
    cout << "Theta: ";
    for(auto param : theta) {
        cout << param << " ";
    }
    cout << endl;
    return 0; 
}