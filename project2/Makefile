target:64_K0 64_K1 64_K2 64_K3 64_K4 256_K0 256_K1 256_K2 256_K3 256_K4

64_K0:histogram64.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram64.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -DK0 -o 64_K0
64_K1:histogram64.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram64.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -DK1 -o 64_K1
64_K2:histogram64.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram64.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -DK2 -o 64_K2
64_K3:histogram64.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram64.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -DK3 -o 64_K3
64_K4:histogram64.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram64.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -DK4 -o 64_K4


256_K0:histogram256.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram256.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -DK0 -o 256_K0
256_K1:histogram256.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram256.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -DK1 -o 256_K1
256_K2:histogram256.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram256.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -DK2 -o 256_K2
256_K3:histogram256.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram256.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -DK3 -o 256_K3
256_K4:histogram256.cu parseOprand.cc histogram_gold.cpp
	nvcc histogram256.cu parseOprand.cc histogram_gold.cpp -Iinc  -O3  -DK4 -o 256_K4
clean:
	rm -rf 64_K0 64_K1 64_K2 64_K3 64_K4 256_K0 256_K1 256_K2 256_K3 256_K4 *.o