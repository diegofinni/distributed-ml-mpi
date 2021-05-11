#include <iostream>
#include <mpi.h>
#include <vector>
using namespace std;

void update_params(vector<double>& params);

void work(vector<double>& params, int num_epoch, int rank);