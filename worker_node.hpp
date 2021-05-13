#include <iostream>
#include <mpi.h>
#include <vector>
using namespace std;

void update_params(vector<double>& params);

void work(vector<double>& params, vector<vector<double> > data_shard, int num_epoch, int rank, int num_workers, string infile);