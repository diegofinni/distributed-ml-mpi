#include <unistd.h>
#include <stdlib.h>
#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <set>
#include <mpi.h>
#include "master_node.hpp"

MPI_Request *reqs, end_req;
vector<double> master_params;
double **worker_params;
set<int> halted_workers;
int *iters;
double lr;
int N, bound, num_workers, num_params, epochs, active_workers;
static const int SLEEP_INTERVAL = 1000;

static inline int rank_to_idx(int rank) { return rank - 1; }

static inline int idx_to_rank(int idx) { return idx + 1; }

void init_master_node(vector<double>& params,
                    int num_nodes,
                    int num_epoch,
                    int n_bound,
                    double learning_rate) {
    // Initialize global variables
    epochs = num_epoch;
    bound = n_bound;
    master_params = params;
    num_workers = num_nodes;
    active_workers = num_workers;
    N = params.size();
    lr = learning_rate;

    // Allocate space for global vectors
    iters = (int*) calloc(num_workers, sizeof(int));
    worker_params = (double**) calloc(num_workers, sizeof(double*));
    for (int i = 0; i < num_workers; i++) {
        worker_params[i] = (double*) calloc(N, sizeof(double));
    }
    reqs = (MPI_Request*) calloc(num_workers, sizeof(MPI_Request));

    // Send out async receive requests to all workers
    for (int idx = 0; idx < num_workers; idx++) {
        int rank = idx_to_rank(idx);
        MPI_Irecv(worker_params[idx], N, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &reqs[idx]);
    }
}

void send_params(int rank) {
    int idx = rank_to_idx(rank);
    MPI_Send(&master_params, N, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
    MPI_Irecv(worker_params[idx], N, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &reqs[idx]);
}

void send_terminations() {
    int a = -1;
    MPI_Ibcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD, &end_req);
}

vector<double> manage_workers() {
    int idx, flag, counter;
    while (1) {
        MPI_Testany(num_workers, reqs, &idx, &flag, MPI_STATUS_IGNORE);
        while (idx != MPI_UNDEFINED) {
            // Grab rank of sender and update its iters
            int rank = idx_to_rank(idx);
            //cout << rank << " completed iteration " << iters[idx] << endl;
            iters[idx]++;
            // Update master_params

            for (int i = 0; i < N; i++) {
                master_params[i] -= lr * worker_params[idx][i];
            }
            if (++counter == epochs) {
                send_terminations();
                return master_params;
            }
            // Find iter of straggler and halt worker if bound has been reached
            const int min = *min_element(iters, iters + num_workers);
            if (iters[idx] - min > bound ) {
                halted_workers.insert(rank);
            }
            // Else send updated parameters to worker
            else {
                send_params(rank);
            }
            // Identify any workers that can now be unhalted
            set<int> unhalted_workers;
            for (set<int>::iterator it = halted_workers.begin(); it != halted_workers.end(); it++) {
                int idx = rank_to_idx(*it);
                if (iters[idx] - min <= bound) {
                    unhalted_workers.insert(*it);
                }
            }
            // Unhalt identified workers and send them new parameters
            for (set<int>::iterator it = unhalted_workers.begin(); it != unhalted_workers.end(); it++) {
                halted_workers.erase(*it);
                send_params(*it);
            }
            MPI_Testany(num_workers, reqs, &idx, &flag, MPI_STATUS_IGNORE);
        }
    }
}
