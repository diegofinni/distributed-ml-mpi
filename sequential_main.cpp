
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
    int num_epoch = stoi(argv[1]);
    double learning_rate = stod(argv[2]);
    string infile = string(argv[3]);
    string outfile = string(argv[4]);
    label1 = string(argv[5]);
    label2 = string(argv[6]);
    /*
    cout << "The input file is: " << infile << endl;
    cout << "The output file is: " << outfile << endl;
    cout << "label1 is: " << "\'" << label1 << "\'" << endl;
    cout << "label2 is: " << "\'" << label2 << "\'" << endl;
    */
    
    if(!parse_flags(argc, argv, 7)) return 0;
    // Data is stored as a 2d vector. Each row is a pair of data, label.
    vector<vector<double> > data = input_data(infile);
    
    init_theta((data[0]).size() - 1);
    init_gradient((data[0]).size() - 1);
    auto t1 = high_resolution_clock::now();
    int m = data.size();
    
    for(int i=0; i<num_epoch; i++)
    {
        train(data, 1);
       
        // update theta
        for(int j=0; j<theta.size(); j++) {
            theta[j] -= learning_rate * gradient[j] / m;
        }
        reset_gradient();
    }

    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as a double. */
    duration<double, milli> ms_double = t2 - t1;

    printf("%.3f\n", ms_double.count() / 1000);

    // Now theta is updated based on training data.
    // Send to other nodes
    /*
    // Debugging: Print theta
    cout << "Theta: ";
    for(auto param : theta) {
        cout << param << " ";
    }
    cout << endl;
    */
    return 0; 
}