#include <iostream>
#include "lib.hpp"
using namespace std;


void init_master_node(vector<double>& params, int num_procs, ReduceFunction func);
    
void manage_workers();