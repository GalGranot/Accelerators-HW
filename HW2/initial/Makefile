DEBUG=0

ifneq ($(DEBUG), 0)
CFLAGS=-O0 -g -G
else
CFLAGS=-O3 -lineinfo
endif

CFLAGS+=-Xcompiler=-Wall -maxrregcount=32 -arch=sm_75
CFLAGS+=`pkg-config opencv4 --cflags --libs`
# Use to find out shared memory size
# CFLAGS+=--ptxas-options=-v 

FILES=ex2 hello-shmem

all: $(FILES)

ex2: ex2.o main.o ex2-cpu.o 
	nvcc --link -L. -lutils $(CFLAGS) $^ -o $@
hello-shmem: hello-shmem.o
	nvcc --link $(CFLAGS) $^ -o $@

ex2.o: ex2.cu ex2.h
main.o: main.cu ex2.h

%.o: %.cu
	nvcc --compile -dc $< $(CFLAGS) -o $@

clean::
	rm -f *.o $(FILES)
