#include <vector>
using namespace std;

void init_master_node(vector<double>& params, 
                    int num_nodes, 
                    int num_epoch, 
                    int n_bound, 
                    double learning_rate);
    
vector<double> manage_workers();