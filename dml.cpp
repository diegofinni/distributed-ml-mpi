#include <cstdlib>
#include <mpi.h>
#include "dml.hpp"

MPI_Request *bcast_reqs;
MPI_Comm *worlds;
int my_rank, num_proc, bound, my_iter, *node_iters;


inline static int rank_to_idx(int rank) {
    return (rank > my_rank) ? rank - 1 : rank;
}

inline static int idx_to_rank(int idx) {
    return (idx >= my_rank) ? idx + 1 : idx;
}

inline static void recv_bcast(int sender_rank) {
    int idx = rank_to_idx(sender_rank);
    MPI_Ibcast(&node_iters[idx], 1, MPI_INT, sender_rank, worlds[sender_rank], &bcast_reqs[idx]);
}

// Return -1 if process must stop due to reaching staleness bound, return 0 otherwise
static int check_bcasts() {
    int idx, flag;
    MPI_Testany(num_proc-1, bcast_reqs, &idx, &flag, MPI_STATUS_IGNORE);
    while (idx != MPI_UNDEFINED) {
        int rank = idx_to_rank(idx);
        int rank_iter = node_iters[idx];
        if ((my_rank - rank_iter) >= bound) return -1;
        MPI_Testany(num_proc-1, bcast_reqs, &idx, &flag, MPI_STATUS_IGNORE);
    }
    return 0;
}

void sum_func(double *dst, const double *src, int n) {
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

ReduceFunction sum_reduce = &sum_func;

void reduce_phase(double *params, int N, int src, int dst, ReduceFunction f) {
    int partition_size = N / num_proc;
    auto *tmp = (double*) malloc(partition_size * sizeof(double));
    // Reduction requires num_proc iterations
    for (int i = 0; i < num_proc - 1; i++) {
        // Calculate indexes of parameter array that will be sent and received
        int send_idx = (my_rank - i) % num_proc;
        send_idx = (send_idx < 0) ? num_proc + send_idx : send_idx;
        int send_start = send_idx * partition_size;
        int send_size = (send_idx == num_proc - 1) ? N - send_start: partition_size;
        
        int recv_idx = (send_idx == 0) ? num_proc - 1 : send_idx - 1;
        int recv_start = recv_idx * partition_size;
        int recv_size = (recv_idx == num_proc - 1) ? N - recv_start : partition_size;

        // Asynchronously send parameter and synchronously receive parameter
        MPI_Request req;
        MPI_Isend(&params[send_start], send_size, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, &req);
        MPI_Recv(tmp, recv_size, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        f(params+recv_start, tmp, recv_size);
    }
}

void share_phase(double *params, int N, int src, int dst) {
    int partition_size = N / num_proc;
    for (int i = 0; i < num_proc - 1; i++) {
        // Calculate indexes of parameter array that will be sent and received
        int send_idx = (my_rank + 1 - i) % num_proc;
        send_idx = (send_idx < 0) ? num_proc + send_idx : send_idx;
        int send_start = send_idx * partition_size;
        int send_size = (send_idx == num_proc - 1) ? N - send_start: partition_size;
        
        int recv_idx = (send_idx == 0) ? num_proc - 1 : send_idx - 1;
        int recv_start = recv_idx * partition_size;
        int recv_size = (recv_idx == num_proc - 1) ? N - recv_start : partition_size;

        // Asynchronously send parameter and synchronously receive parameter
        MPI_Request req;
        MPI_Isend(&params[send_start], send_size, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, &req);
        MPI_Recv(&params[recv_start], recv_size, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void reduce(double *params, int N, ReduceFunction f) {
    // Compute my_ranks of neighbors that node will communicate with
    int src = (my_rank == 0) ? num_proc - 1 : my_rank - 1;
    int dst = (my_rank == num_proc - 1) ? 0 : my_rank + 1;

    reduce_phase(params, N, src, dst, f);
    share_phase(params, N, src, dst);
}

void init_mpi_env(int rank, int num_procs, int stale_bound) {
    my_rank = rank;
    num_proc = num_procs;
    bound = stale_bound;

    bcast_reqs = (MPI_Request*) malloc((num_proc-1) * sizeof(MPI_Request));
    worlds = (MPI_Comm*) malloc(num_proc * sizeof(MPI_Comm));
    node_iters = (int*) calloc(num_proc - 1, sizeof(int));

    for (int i = 0; i < num_proc; i++) {
        MPI_Comm_dup(MPI_COMM_WORLD, &worlds[i]);
        if (i == my_rank) continue;
        recv_bcast(i);
    }
}