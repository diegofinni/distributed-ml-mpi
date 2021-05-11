#include <vector>
#include <string>
using namespace std;

extern vector<double> theta;
extern vector<double> gradient;
extern string label1;
extern string label2;

// TODO: CUDA optimizations (will probably use Thrust, CUDA's C++ library)
extern int CUDA_ENABLE;
// Optional: Sparse vectors
extern int SPARSE_ENABLE;

/*
 * input_data parses a csv file data into a 2d vector. Each row has vector data followed by the label (0 or 1).
 *
 * No column of training data can be non-numeric (delete column)
 */
vector<vector<double> > input_data(string infile);

/*
 * init_theta allocates space for theta, our parameter vector
 */
void init_theta(int num_features);

/*
 * init_theta allocates space for gradient, our gradient vector
 */
void init_gradient(int num_features);

/*
 * reset_gradient sets our gradient vector to 0
 */
void reset_gradient();

/*
 * dotprod takes the dotprod of v1 and v2, 
 * using the size of v2 (for functionality despite bias term)
 */
double ml_dotprod(vector<double> params, vector<double> data);

/* 
 * sigmoid bounds x to 0 or 1 (our prediction)
 */
double sigmoid(double x);

/*
 * SGD_step edits our theta parameter vector based on a single data row.
 * 
 * Loss function: cross-entropy loss function
 */
void SGD_step(vector<double> data_row, double learning_rate);

/* 
 * train learns appropriate theta based on training data
*/
void train(vector<vector<double> > data, double learning_rate, int num_epoch);

void print_usage();

/*
 * parse_flags parses options inputted from CMD 
 * returns 0 if we do not want to run, 1 otherwise
 */
int parse_flags(int argc, char **argv, int offset);