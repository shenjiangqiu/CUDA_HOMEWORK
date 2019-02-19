#author Jiangqiu Shen
#email jshen2@mtu.edu


.PHONY:all
BUILDPATH=build
SRCPATH=src
CUFILE=src/TiledMatrixMul.cu
CCFILE=src/TiledMatrixMul.cc src/parseOprand.cc
CCOBJ := $(patsubst $(SRCPATH)/%.cc, $(BUILDPATH)/%.cc_o, $(CCFILE))


CUOBJ := $(patsubst $(SRCPATH)/%.cu, $(BUILDPATH)/%.cu_o, $(CUFILE))


CUFLAGS = -O3
CCFLAGS = -O3 -std=c++14 -Wall  
LIB = -L/opt/cuda/lib64/ -lcudart -lpthread
INC=-I/opt/cuda/include -I../inc -Iinc

all:$(BUILDPATH)/TiledMatrixMul $(BUILDPATH)/NormalMatrixMul

$(BUILDPATH)/TiledMatrixMul:$(BUILDPATH)/TiledMatrixMul.cu_o $(BUILDPATH)/TiledMatrixMul.cc_o $(BUILDPATH)/parseOprand.cc_o
	g++  $(CCFLAGS) $^ $(LIB) -o $@;

$(BUILDPATH)/NormalMatrixMul:

$(CUOBJ): $(BUILDPATH)/%.cu_o:src/%.cu 
	nvcc $(CUFLAGS) $(INC) $^ -c  -o $@


$(CCOBJ): $(BUILDPATH)/%.cc_o:src/%.cc 
	g++ $(CCFLAGS) $(INC) $^ -c  -o $@


.PHONY:clean
clean:
	rm -rf $(BUILDPATH)/*