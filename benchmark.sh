./lr_seq 2000 0.01 data/diabetes2.csv out.txt 0 1

mpirun -np 1 ./lr_decentralized 2000 0.01 data/diabetes2.csv out.txt 0 1
mpirun -np 2 ./lr_decentralized 2000 0.01 data/diabetes2.csv out.txt 0 1
mpirun -np 3 ./lr_decentralized 2000 0.01 data/diabetes2.csv out.txt 0 1
mpirun -np 4 ./lr_decentralized 2000 0.01 data/diabetes2.csv out.txt 0 1
mpirun -np 5 ./lr_decentralized 2000 0.01 data/diabetes2.csv out.txt 0 1
mpirun -np 6 ./lr_decentralized 2000 0.01 data/diabetes2.csv out.txt 0 1

mpirun -np 2 ./lr_centralized 100 0.01 5 10 data/bagofwords.csv out.txt 0 1
mpirun -np 3 ./lr_centralized 100 0.01 5 10 data/bagofwords.csv out.txt 0 1
mpirun -np 4 ./lr_centralized 100 0.01 5 10 data/bagofwords.csv out.txt 0 1
mpirun -np 5 ./lr_centralized 100 0.01 5 10 data/bagofwords.csv out.txt 0 1
mpirun -np 6 ./lr_centralized 100 0.01 5 10 data/bagofwords.csv out.txt 0 1
mpirun -np 7 ./lr_centralized 100 0.01 5 10 data/bagofwords.csv out.txt 0 1
mpirun -np 8 ./lr_centralized 100 0.01 5 10 data/bagofwords.csv out.txt 0 1