target=64_k0 64_k1 64_k2 64_k3 64_k4 256_k0 256_k1 256_k2 256_k3 256_k4

64_k0:histogram64.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram64.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -DK0 -o 64_k0
64_k1:histogram64.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram64.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -Dk1 -o 64_k1
64_k2:histogram64.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram64.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -Dk2 -o 64_k2
64_k3:histogram64.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram64.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -Dk3 -o 64_k3
64_k4:histogram64.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram64.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -Dk4 -o 64_k4


256_k0:histogram256.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram256.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -DK0 -o 256_k0
256_k1:histogram256.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram256.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -Dk1 -o 256_k1
256_k2:histogram256.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram256.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -Dk2 -o 256_k2
256_k3:histogram256.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram256.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -Dk3 -o 256_k3
256_k4:histogram256.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram256.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -Dk4 -o 256_k4
clean:
	rm -rf 64_k0 64_k1 64_k2 64_k3 64_k4 256_k0 256_k1 256_k2 256_k3 256_k4 *.o