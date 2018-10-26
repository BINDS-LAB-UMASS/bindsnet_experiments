
NVCC           :="/usr/local/cuda-8.0/bin/nvcc"
NVCCFLAGS      :=-std=c++11 -x cu -arch sm_52 -O3

INCLUDEFLAGS   =-I"$(GENN_PATH)/lib/include"

all: runner.o

-include runner.d
runner.d: runner.cc
	$(NVCC) -M $(NVCCFLAGS) $(INCLUDEFLAGS) runner.cc 1> runner.d

runner.o: runner.cc runner.d
	$(NVCC) -c $(NVCCFLAGS) $(INCLUDEFLAGS) runner.cc

clean:
	rm -f runner.o runner.d
