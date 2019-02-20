#author Jiangqiu Shen
#email jshen2@mtu.edu


.PHONY:all
#DEFINE='-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG'
BUILDPATH=build
SRCPATH=src
CUFILE=src/TiledMatrixMul.cu 
CCFILE= src/parseOprand.cc
CCOBJ := $(patsubst $(SRCPATH)/%.cc, $(BUILDPATH)/%.cc_o, $(CCFILE))


CUOBJ := $(patsubst $(SRCPATH)/%.cu, $(BUILDPATH)/%.cu_o, $(CUFILE))


CUFLAGS = -O3 $(DEFINE)  -std=c++14
CCFLAGS = -O3 -std=c++14 -Wall  $(DEFINE)
LIB = -L/opt/cuda/lib64/ -lcudart -lpthread
INC=-I/opt/cuda/include -I../inc -Iinc -Isrc

all:$(BUILDPATH)/TiledMatrixMul 
src/TiledMatrixMul.cu:src/TiledMatrixMul_kernel.cu
	touch $@
$(BUILDPATH)/TiledMatrixMul:$(BUILDPATH)/TiledMatrixMul.cu_o  $(BUILDPATH)/parseOprand.cc_o
	g++  $(CCFLAGS) $^ $(LIB) -o $@;


$(CUOBJ): $(BUILDPATH)/%.cu_o:src/%.cu 
	nvcc $(CUFLAGS) $(INC) $^ -c  -o $@


$(CCOBJ): $(BUILDPATH)/%.cc_o:src/%.cc 
	g++ $(CCFLAGS) $(INC) $^ -c  -o $@


.PHONY:clean
clean:
	rm -rf $(BUILDPATH)/*