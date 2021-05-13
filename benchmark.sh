mpirun -np 2 ./lr_decentralized 1000 0.01 data/weatherAUS.csv out.txt Yes No
mpirun -np 3 ./lr_decentralized 1000 0.01 data/weatherAUS.csv out.txt Yes No
mpirun -np 4 ./lr_decentralized 1000 0.01 data/weatherAUS.csv out.txt Yes No
mpirun -np 5 ./lr_decentralized 1000 0.01 data/weatherAUS.csv out.txt Yes No
mpirun -np 6 ./lr_decentralized 1000 0.01 data/weatherAUS.csv out.txt Yes No
mpirun -np 7 ./lr_decentralized 1000 0.01 data/weatherAUS.csv out.txt Yes No
mpirun -np 8 ./lr_decentralized 1000 0.01 data/weatherAUS.csv out.txt Yes No

mpirun -np 2 ./lr_centralized 1000 0.01 10 data/weatherAUS.csv out.txt Yes No
mpirun -np 3 ./lr_centralized 1000 0.01 10 data/weatherAUS.csv out.txt Yes No
mpirun -np 4 ./lr_centralized 1000 0.01 10 data/weatherAUS.csv out.txt Yes No
mpirun -np 5 ./lr_centralized 1000 0.01 10 data/weatherAUS.csv out.txt Yes No
mpirun -np 6 ./lr_centralized 1000 0.01 10 data/weatherAUS.csv out.txt Yes No
mpirun -np 7 ./lr_centralized 1000 0.01 10 data/weatherAUS.csv out.txt Yes No
mpirun -np 8 ./lr_centralized 1000 0.01 10 data/weatherAUS.csv out.txt Yes No