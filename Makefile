CC=mpic++

CFLAGS= -c -Wall

all: prog

prog: main.o dml.o
	$(CC) main.o dml.o -o dml

main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp

dml.o: dml.cpp
	$(CC) $(CFLAGS) dml.cpp

clean:
	rm -rf *.o 