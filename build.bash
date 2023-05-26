set -e

mkdir -p build
rm -f build/*.o
rm -f build/tensor_test

g++ -Wall -Werror -c tensor_cpu.cpp -o build/tensor_cpu.o
g++ -Wall -Werror -c -I/usr/local/cuda-12.1/include tensor.cpp -o build/tensor.o

# 2080ti has turing architecture. needs arch=compute_50 and sm_50
nvcc --cudart static --relocatable-device-code=false \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_50,code=compute_50 \
    -link -c -o build/tensor_gpu.o tensor_gpu.cu

g++ -Wall -Werror \
    build/tensor.o \
    build/tensor_gpu.o \
    build/tensor_cpu.o \
    tensor_test.cpp \
    -o build/tensor_test -L/usr/local/cuda-12.1/lib64 -lcudart