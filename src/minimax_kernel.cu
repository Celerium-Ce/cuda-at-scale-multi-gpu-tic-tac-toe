// Copyright 2026 Celerium-Ce
#include "include/minimax_player.cuh"

#include <cstdint>

#include "include/board.h"
#include "include/cuda_utils.h"

namespace {

// Recursively scores `board` from the point of view of `me`: positive means
// `me` is winning, negative means `opponent` is winning, magnitude favors a
// faster win / slower loss via the `depth` term. `my_turn` says whose move
// is being placed at this level of the recursion. Tic-tac-toe's game tree
// has at most 9 plies, so this recursion (and its per-call stack copy of
// the board) stays shallow enough to run directly on the GPU without a
// manual stack.
__device__ int MinimaxScore(int8_t board[kBoardCells], int depth,
                             bool my_turn, int8_t me, int8_t opponent) {
  const int8_t winner = CheckWinner(board);
  if (winner == me) {
    return 10 - depth;
  }
  if (winner == opponent) {
    return depth - 10;
  }
  if (IsBoardFull(board)) {
    return 0;
  }

  const int8_t mover = my_turn ? me : opponent;
  int best = my_turn ? -100 : 100;
  for (int i = 0; i < kBoardCells; ++i) {
    if (board[i] != kEmpty) {
      continue;
    }
    board[i] = mover;
    const int score = MinimaxScore(board, depth + 1, !my_turn, me, opponent);
    board[i] = kEmpty;
    if (my_turn ? (score > best) : (score < best)) {
      best = score;
    }
  }
  return best;
}

// One thread per candidate move: thread `tid` plays `me` on
// `empty_cells[tid]`, then exhaustively searches the remaining game (where
// it is now `opponent`'s turn) to produce that candidate's final score.
__global__ void MinimaxMoveKernel(const int8_t* __restrict__ board_in,
                                   const int* __restrict__ empty_cells,
                                   int num_empty, int8_t me, int8_t opponent,
                                   int* __restrict__ scores_out) {
  const int tid = threadIdx.x;
  if (tid >= num_empty) {
    return;
  }

  int8_t local_board[kBoardCells];
  for (int i = 0; i < kBoardCells; ++i) {
    local_board[i] = board_in[i];
  }

  const int candidate_cell = empty_cells[tid];
  local_board[candidate_cell] = me;
  scores_out[tid] = MinimaxScore(local_board, 1, false, me, opponent);
}

}  // namespace

void LaunchMinimaxMove(const int8_t* h_board, int8_t me, int8_t opponent,
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

  MinimaxMoveKernel<<<1, num_empty>>>(d_board, d_empty_cells, num_empty, me,
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
