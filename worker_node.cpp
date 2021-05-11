#include "worker_node.hpp"


void update_params(vector<double>& params) {
    MPI_Send(&params, params.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    MPI_Recv(&params, params.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void work(vector<double>& params, int num_epoch, int rank) {
    for (int i = 0; i < num_epoch; i++) {
        fill(params.begin(), params.end(), (i + 1) * rank);
        cout << rank << " on epoch " << i << ": " << (i + 1) * rank << endl;
        update_params(params);
        cout << rank << " on epoch" << i << ": " << params[0] << endl;
    }
}