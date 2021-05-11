/*
 * linear-regression.cpp is an simple ML algorithm to make predictions based on a linear regression model.
 * 
 * Input data is stored as a 2d vector. Each row is vector<double> data, where the last element is the label.
 * We have to delete data that is not numeric.
 * 
 * There is timing of exectution enabled for testing purposes.
 */

#include <iostream>
#include <string>
#include <fstream>
#include <sstream> //istringstream
#include <fstream> // ifstream
#include <algorithm>
#include <math.h>
#include <chrono>
#include "lr.hpp"

// Timing 
using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::duration;
using chrono::milliseconds;

vector<double> theta;
vector<double> gradient;
string label1;
string label2;
int CUDA_ENABLE = 0;
int SPARSE_ENABLE = 0;

/*
 * input_data parses a csv file data into a 2d vector. Each row has vector data followed by the label (0 or 1).
 *
 * No column of training data can be non-numeric (delete column)
 */
vector<vector<double> > input_data(string infile) {
    vector<vector<double> > data;
    ifstream input_file(infile);
    int row = 0;
    int num_cols = 0;
    while (input_file) {
        string row_string;
        if (!getline(input_file, row_string)) break;
        istringstream ss(row_string);
        vector<double> record;
        bool valid_row = true;
        if(row == 0) {
            num_cols = count(row_string.begin(), row_string.end(), ',') + 1;
        }
        if(row > 0) {
            int col = 0;
            double label;
            // Loop through each col
            while (ss) {
                string cell;
                if (!getline(ss, cell, ',')) break;
                // Data case
                if(col < num_cols-1) {
                    try {
                        size_t idx;
                        double cell_double = stof(cell, &idx);
                        if(idx == cell.length()) {
                            record.push_back(cell_double);
                        }
                    }
                    catch (const invalid_argument e) {
                        valid_row = false;
                        break;
                    }
                    catch (const out_of_range e) {
                        valid_row = false;
                        break;
                    }
                }
                // Label case
                else {
                    // Use find instead of includes because of strange string escape sequences
                    if(cell.find(label1) != std::string::npos) {
                        label = 0;
                    }
                    else if(cell.find(label2) != std::string::npos) {
                        label = 1;
                    }
                    else {
                        valid_row = false;
                        break;
                    }
                    record.push_back(label);
                }
                col += 1;
            }
            if(valid_row) {
                data.push_back(record);
            }
        }
        row++;
    }
    return data;
}

/*
 * init_theta allocates space for theta, our parameter vector
 */
void init_theta(int num_features) {
    //theta.resize(num_features);
    for(int i=0; i<num_features+1; i++) {
        // We include an extra parameter for bias term
        theta.push_back(0);
    }
}

/*
 * init_theta allocates space for gradient, our gradient vector
 */
void init_gradient(int num_features) {
    //theta.resize(num_features);
    for(int i=0; i<num_features+1; i++) {
        // We include an extra parameter for bias term
        gradient.push_back(0);
    }
}

/*
 * reset_gradient sets our gradient vector to 0
 */
void reset_gradient() {
    for(int i=0; i<gradient.size(); i++) {
        // We include an extra parameter for bias term
        gradient[i] = 0;
    }
}


/*
 * dotprod takes the dotprod of v1 and v2, 
 * using the size of v2 (for functionality despite bias term)
 */
double ml_dotprod(vector<double> params, vector<double> data) {
    double result = 0;
    // We don't include the last element of data, as it is the label
    for(int i=0; i<data.size()-1; i++) {
        result += params[i]*data[i];
    }
    return result;
}

/* 
 * sigmoid bounds x to 0 or 1 (our prediction)
 */
double sigmoid(double x) {
    if(x >= 0) {
        double z = exp(-x);
        return 1 / (1 + z);
    }
    else {
        double z = exp(x);
        return z / (1 + z);
    }
}

/*
 * SGD_step edits our theta parameter vector based on a single data row.
 * 
 * Loss function: cross-entropy loss function
 */
void SGD_step(vector<double> data_row, double learning_rate) {
    /*
    // Debugging: Print theta
    cout << "Theta: ";
    for(auto param : theta) {
        cout << param << " ";
    }
    cout << endl;
    */
    
    double bias = theta.back();
    double label = data_row.back();
    double theta_dot_x = ml_dotprod(theta, data_row);
    //cout << "Dot Prod: " << theta_dot_x << endl;
    double sig = sigmoid(theta_dot_x + bias);
    //cout << "Sig: " << sig << " Label: " << label << endl;

    double err = (sig - label);
    /*
    if(err < 0.1) {
        cout << "Correct" << endl;
    }
    else {
        cout << "Wrong" << endl;
    }
    */
    
    for(int i=0; i<gradient.size()-1; i++) {
        gradient[i] += err * data_row[i];
    }
    gradient.back() += err;
}

/* 
 * train learns appropriate theta based on training data
*/
void train(vector<vector<double> > data, double learning_rate, int num_epoch) {
    // Debug: print the data vector
    int rows = data.size();
    for(int i=0; i<num_epoch; i++) {
        for(int j=0; j<rows; j++) {
            SGD_step(data[j], learning_rate);
        }
    }
}

void print_usage() {
    cout << "Usage: ./linear-regression [infile] [outfile] [label1] [label2] [-h for help] [-c to enable CUDA]\n";
}

/*
 * parse_flags parses options inputted from CMD 
 * returns 0 if we do not want to run, 1 otherwise
 */
int parse_flags(int argc, char **argv, int offset) {
    int i = offset;
    while(i < argc) {
        if(argv[i][0] == '-') {
            switch(argv[i][1])
            {
                case 'h': 
                    print_usage();
                    return 0;
                case 'c': 
                    CUDA_ENABLE = 1;
                    cout << "CUDA not implemented yet";
                    return 0;
                case 's':
                    SPARSE_ENABLE = 1;
                    cout << "Sparse vectors not implemented yet";
                    return 0;
                default:
                    cout << "Not a valid flag: -" << argv[i][1] << '\n';
            }
        }
    }
    return 1;
}