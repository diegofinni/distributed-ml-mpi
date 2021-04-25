#include <stdlib.h>

#include "mpi.h"
#include "dml.h"

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

void *intSumFunc(void *dst, void *src, int n) {
    int *intDst = (int *) dst;
    int *intSrc = (int *) src;
    for (int i = 0; i < n; i++) {
        intDst[i] += intSrc[i];
    }
    return (void*)intDst;
}

void *floatSumFunc(void *dst, void *src, int n) {
    float *floatDst = (float *) dst;
    float *floatSrc = (float *) src;
    for (int i = 0; i < n; i++) {
        floatDst[i] += floatSrc[i];
    }
    return (void*)floatDst;
}

ReduceFunction intSum = &intSumFunc;
ReduceFunction floatSum = &floatSumFunc;

void reducePhase(int *params, int N, int src, int dst, ReduceFunction f) {
    int partitionSize = N / numProc;
    int *tmp = malloc(partitionSize * sizeof(int));
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
        MPI_Isend(&params[sendStart], sendSize, MPI_INT, dst, 0, MPI_COMM_WORLD, &req);
        MPI_Recv(tmp, recvSize, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        f(params+recvStart, tmp, recvSize);
    }
}

void sharePhase(int *params, int N, int src, int dst) {
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
        MPI_Isend(&params[sendStart], sendSize, MPI_INT, dst, 0, MPI_COMM_WORLD, &req);
        MPI_Recv(&params[recvStart], recvSize, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void ringAllReduce(int *params, int N, ReduceFunction f) {
    // Compute ranks of neighbors that node will communicate with
    int src = (rank == 0) ? numProc - 1 : rank - 1;
    int dst = (rank == numProc - 1) ? 0 : rank + 1;

    reducePhase(params, N, src, dst, f);
    sharePhase(params, N, src, dst);
}