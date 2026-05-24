# GPU Auction Algorithm Makefile
CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := -arch=sm_89
OPT       := -O3
TARGET    := auction_cuda.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): auction_kernel.cu
	$(NVCC) -shared $(ARCH) $(OPT) -Xcompiler -fPIC \
		-o $(TARGET) auction_kernel.cu \
		--compiler-options -Wall

clean:
	rm -f $(TARGET)

test: $(TARGET)
	python3 -c "import ctypes; lib = ctypes.CDLL('./$(TARGET)'); print('Loaded OK:', lib)"
