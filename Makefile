# CUDA Multi-GPU Tic-Tac-Toe
# Builds bin/gpu_tictactoe: two CUDA kernels (minimax search, heuristic
# scoring) competing turn-by-turn as Player X and Player O.
#
# Requirements: CUDA Toolkit 10.1+ (nvcc on PATH) and at least one
# CUDA-capable GPU (a second GPU lets the two players run on separate
# physical devices; one GPU still works, see README).
#
# Usage:
#   make                 # build bin/gpu_tictactoe
#   make run             # build, then play one game, logging to logs/
#   make clean           # remove build artifacts
#
# Override the target GPU architecture(s) if the default list below does
# not cover your device, e.g.:
#   make ARCHES="-gencode arch=compute_86,code=sm_86"

NVCC      ?= nvcc

TARGET    := bin/gpu_tictactoe
SRC_DIR   := src
OBJ_DIR   := bin/obj

CPP_SRCS  := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS   := $(wildcard $(SRC_DIR)/*.cu)
OBJS      := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS)) \
             $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))

# -I. lets "include/foo.h" in src/*.cpp|cu resolve from the project root
# (Google C++ Style Guide: include paths mirror the directory layout).
NVCCFLAGS := -std=c++17 -O2 -I.
LDFLAGS   := -lcudart

# Covers Maxwell (sm_52) through Ampere (sm_86); adjust for newer/older GPUs.
ARCHES ?= -gencode arch=compute_52,code=sm_52 \
          -gencode arch=compute_60,code=sm_60 \
          -gencode arch=compute_70,code=sm_70 \
          -gencode arch=compute_75,code=sm_75 \
          -gencode arch=compute_80,code=sm_80 \
          -gencode arch=compute_86,code=sm_86

.PHONY: all dirs run clean

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJ_DIR) bin logs

$(TARGET): $(OBJS)
	$(NVCC) $(ARCHES) $^ -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | dirs
	$(NVCC) $(NVCCFLAGS) $(ARCHES) -c $< -o $@

run: all
	./$(TARGET) --log logs/game.log

clean:
	rm -rf $(OBJ_DIR) $(TARGET)
