/*
 * linear-regression.cpp is an simple ML algorithm to make predictions based on a linear regression model.
 * 
 * Input data is stored as a 2d vector. Each row is a pair of vector<double> data and double label (0 or 1).
 * We delete data  that is not numeric.
 * 
 * 
 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream> //istringstream
#include <fstream> // ifstream
#include <algorithm>
#include <math.h>
using namespace std;

// TODO: CUDA optimizations (will probably use Thrust, CUDA's C++ library)
int CUDA_ENABLE = 0;
// Optional: Sparse vectors
int SPARSE_ENABLE = 0;
vector<double> theta;
string label1;
string label2;

/*
 * input_data parses a csv file data into a 2d vector. Each row has vector data followed by the label (0 or 1).
 *
 * No column of training data can be non-numeric (delete column)
 */
vector<pair<vector<double>,double> > input_data(string infile) {
    vector<pair<vector<double>,double> > data;
    ifstream input_file(infile);
    int row, num_cols = 0;
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
                cout << cell << " ";
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
                }
                col += 1;
            }
            if(valid_row) {
                data.push_back(make_pair(record,label));
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
    cout << "PUSHED BACK: " << num_features+1 << endl;
}

/*
 * dotprod takes the dotprod of v1 and v2, 
 * using the size of v2 (for functionality despite bias term)
 */
double dotprod(vector<double> v1, vector<double> v2) {
    double result = 0;
    for(int i=0; i<v2.size(); i++) {
        result += v1[i]*v2[i];
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
void SGD_step(pair<vector<double>,double> data_row, double learning_rate) {
    /*
    // Debugging: Print theta
    cout << "Theta: ";
    for(auto param : theta) {
        cout << param << " ";
    }
    cout << endl;
    */
    
    vector<double> data_vals = data_row.first;
    double bias = theta.back();
    double label = data_row.second;
    double theta_dot_x = dotprod(theta, data_vals);
    //cout << "Dot Prod: " << theta_dot_x << endl;
    double sig = sigmoid(theta_dot_x + bias);
    //cout << "Sig: " << sig << " Label: " << label << endl;
    double gradient = (sig - label);
    for(int i=0; i<theta.size()-1; i++) {
        theta[i] -= learning_rate * gradient * data_vals[i];
    }
    theta.back() -= learning_rate * gradient;
}

/* 
 * train learns appropriate theta based on training data
*/
void train(vector<pair<vector<double>,double> > data, double learning_rate, int num_epoch) {
    int rows = data.size();
    for(int i=0; i<num_epoch; i++) {
        for(int j=0; j<rows; j++) {
            vector<double> data_vals = data[j].first;
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
    vector<pair<vector<double>,double> > data = input_data(infile);
    
    /*
    // Debug: print the data vector
    for(auto row : data) {
        cout << "Data: ";
        for(auto cell : row.first) {
            cout << cell << " ";
        }
        cout << endl;
        cout << "Label: " << row.second << endl;
    }
    */
    init_theta((data[0].first).size());

    double learning_rate = 0.01;
    int num_epoch = 1;
    train(data, learning_rate, num_epoch);

    // Now theta is updated based on training data.
    // Send to other nodes

    return 0; 
}