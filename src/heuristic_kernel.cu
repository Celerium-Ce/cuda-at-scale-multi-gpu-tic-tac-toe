// Copyright 2026 Celerium-Ce
#include "include/heuristic_player.cuh"

#include <cstdint>

#include "include/board.h"
#include "include/cuda_utils.h"

namespace {

// True if cell `i` is one of the board's 4 corners.
__device__ __forceinline__ bool IsCorner(int i) {
  return i == 0 || i == 2 || i == 6 || i == 8;
}

// Scores one candidate cell for `me`, with no search beyond one ply:
// an immediate win beats an immediate block, which beats the center,
// which beats a corner, which beats an edge. This is intentionally a
// shallower strategy than the full minimax search in minimax_kernel.cu.
__device__ int ScoreCandidate(const int8_t* board_in, int cell, int8_t me,
                               int8_t opponent) {
  int8_t probe[kBoardCells];
  for (int i = 0; i < kBoardCells; ++i) {
    probe[i] = board_in[i];
  }

  probe[cell] = me;
  if (CheckWinner(probe) == me) {
    return 100;
  }

  probe[cell] = opponent;
  if (CheckWinner(probe) == opponent) {
    return 50;
  }

  if (cell == 4) {
    return 10;
  }
  if (IsCorner(cell)) {
    return 5;
  }
  return 1;
}

// One thread per candidate move: thread `tid` scores `empty_cells[tid]`
// independently of every other thread, so all candidates are evaluated in
// parallel rather than in a loop.
__global__ void HeuristicMoveKernel(const int8_t* __restrict__ board_in,
                                     const int* __restrict__ empty_cells,
                                     int num_empty, int8_t me,
                                     int8_t opponent,
                                     int* __restrict__ scores_out) {
  const int tid = threadIdx.x;
  if (tid >= num_empty) {
    return;
  }
  scores_out[tid] = ScoreCandidate(board_in, empty_cells[tid], me, opponent);
}

}  // namespace

void LaunchHeuristicMove(const int8_t* h_board, int8_t me, int8_t opponent,
                          int device_id, int* out_move) {
  CUDA_CHECK(cudaSetDevice(device_id));

  int empty_cells[kBoardCells];
  const int num_empty = CollectEmptyCells(h_board, empty_cells);

  int8_t* d_board = nullptr;
  int* d_empty_cells = nullptr;
  int* d_scores = nullptr;
  CUDA_CHECK(cudaMalloc(&d_board, kBoardCells * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_empty_cells, num_empty * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_scores, num_empty * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_board, h_board, kBoardCells * sizeof(int8_t),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_empty_cells, empty_cells, num_empty * sizeof(int),
                         cudaMemcpyHostToDevice));

  HeuristicMoveKernel<<<1, num_empty>>>(d_board, d_empty_cells, num_empty, me,
                                         opponent, d_scores);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int scores[kBoardCells];
  CUDA_CHECK(cudaMemcpy(scores, d_scores, num_empty * sizeof(int),
                         cudaMemcpyDeviceToHost));

  int best_index = 0;
  for (int i = 1; i < num_empty; ++i) {
    if (scores[i] > scores[best_index]) {
      best_index = i;
    }
  }
  *out_move = empty_cells[best_index];

  CUDA_CHECK(cudaFree(d_board));
  CUDA_CHECK(cudaFree(d_empty_cells));
  CUDA_CHECK(cudaFree(d_scores));
}
