# GPU Auction Algorithm Makefile
CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc
ARCH      := -arch=sm_89
OPT       := -O3
TARGET    := auction_cuda.so
EDGE_TGT  := edge_builder.so

.PHONY: all clean

all: $(TARGET) $(EDGE_TGT)

$(TARGET): auction_kernel.cu
	$(NVCC) -shared $(ARCH) $(OPT) -Xcompiler -fPIC \
		-o $(TARGET) auction_kernel.cu \
		--compiler-options -Wall

$(EDGE_TGT): edge_builder.cu
	$(NVCC) -shared $(ARCH) $(OPT) -Xcompiler -fPIC \
		-o $(EDGE_TGT) edge_builder.cu \
		--compiler-options -Wall

clean:
	rm -f $(TARGET) $(EDGE_TGT)

test: $(TARGET)
	python3 -c "import ctypes; lib = ctypes.CDLL('./$(TARGET)'); print('Loaded OK:', lib)"
