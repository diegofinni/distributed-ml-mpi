#include <unistd.h>
#include <mpi.h>
#include "master_node.hpp"

ReduceFunction f;
MPI_Request *reqs;
vector<double> master_params;
int N, num_workers, num_params;
static const int SLEEP_INTERVAL = 100;

static inline int rank_to_idx(int rank) { return rank - 1; }

static inline int idx_to_rank(int idx) { return idx + 1; }

void init_master_node(vector<double>& params, int num_procs, ReduceFunction func) {
    // Initialize global variables
    master_params = params;
    num_workers = num_procs - 1;
    N = params.size();
    f = func;

    // Allocate space for global vectors
    worker_params.reserve(num_workers);
    for (int i = 0; i < num_workers; i++) worker_params[i].reserve(N);
    reqs.reserve(num_workers);

    // Send out async receive requests to all workers
    for (int idx = 0; idx < num_workers; idx++) {
        int rank = idx_to_rank(idx);
        MPI_Irecv(&worker_params[idx], N, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &reqs[idx]);
    }
}

void manage_workers() {
    int idx, flag;
    while(1) {
        usleep(SLEEP_INTERVAL);
        MPI_Testany(num_workers, reqs, &idx, &flag, MPI_STATUS_IGNORE);
        while (idx != MPI_UNDEFINED) {
            int rank = idx_to_rank(idx);

        }
    }
}