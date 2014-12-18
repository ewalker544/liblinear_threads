CXX ?= g++
CC = gcc
#CFLAGS = -Wall -Wconversion -O3 -fPIC
CFLAGS = -Wall -Wconversion -O3
CXXFLAGS = -std=c++11 -Wall -Wconversion -O3
LIBS = blas/blas.a
SHVER = 2
OS = $(shell uname)
#LIBS = -lblas

all: train-par predict-par

train-par: tron.o linear_par.o train.c blas/blas.a
	$(CXX) $(CXXFLAGS) -o train-par train.c tron.o linear_par.o $(LIBS)

predict-par: tron.o linear_par.o predict.c blas/blas.a
	$(CXX) $(CXXFLAGS) -o predict-par predict.c tron.o linear_par.o $(LIBS)

tron.o: tron.cpp tron.h
	$(CXX) $(CXXFLAGS) -c -o tron.o tron.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	rm -f *~ tron.o linear_par.o train-par predict-par
