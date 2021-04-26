#include <stdlib.h>

#include <mpi.h>
#include "dml.hpp"

void sumFunc(double *dst, const double *src, int n) {
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

ReduceFunction sumReduce = &sumFunc;

void reducePhase(double *params, int N, int src, int dst, ReduceFunction f) {
    int partitionSize = N / numProc;
    auto *tmp = (double*) malloc(partitionSize * sizeof(double));
    // Reduction requires numProc iterations
    for (int i = 0; i < numProc - 1; i++) {
        // Calculate indexes of parameter array that will be sent and received
        int sendIdx = (rank - i) % numProc;
        sendIdx = (sendIdx < 0) ? numProc + sendIdx : sendIdx;
        int sendStart = sendIdx * partitionSize;
        int sendSize = (sendIdx == numProc - 1) ? N - sendStart: partitionSize;
        
        int recvIdx = (sendIdx == 0) ? numProc - 1 : sendIdx - 1;
        int recvStart = recvIdx * partitionSize;
        int recvSize = (recvIdx == numProc - 1) ? N - recvStart : partitionSize;

        // Asynchronously send parameter and synchronously receive parameter
        MPI_Request req;
        MPI_Isend(&params[sendStart], sendSize, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, &req);
        MPI_Recv(tmp, recvSize, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        f(params+recvStart, tmp, recvSize);
    }
}

void sharePhase(double *params, int N, int src, int dst) {
    int partitionSize = N / numProc;
    for (int i = 0; i < numProc - 1; i++) {
        // Calculate indexes of parameter array that will be sent and received
        int sendIdx = (rank + 1 - i) % numProc;
        sendIdx = (sendIdx < 0) ? numProc + sendIdx : sendIdx;
        int sendStart = sendIdx * partitionSize;
        int sendSize = (sendIdx == numProc - 1) ? N - sendStart: partitionSize;
        
        int recvIdx = (sendIdx == 0) ? numProc - 1 : sendIdx - 1;
        int recvStart = recvIdx * partitionSize;
        int recvSize = (recvIdx == numProc - 1) ? N - recvStart : partitionSize;

        // Asynchronously send parameter and synchronously receive parameter
        MPI_Request req;
        MPI_Isend(&params[sendStart], sendSize, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, &req);
        MPI_Recv(&params[recvStart], recvSize, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void ringAllReduce(double *params, int N, ReduceFunction f) {
    // Compute ranks of neighbors that node will communicate with
    int src = (rank == 0) ? numProc - 1 : rank - 1;
    int dst = (rank == numProc - 1) ? 0 : rank + 1;

    reducePhase(params, N, src, dst, f);
    sharePhase(params, N, src, dst);
}