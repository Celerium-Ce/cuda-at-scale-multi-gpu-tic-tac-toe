// Copyright 2026 Celerium-Ce
#ifndef INCLUDE_BOARD_H_
#define INCLUDE_BOARD_H_

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

// A tic-tac-toe board is 9 cells, row-major, index = row * 3 + col:
//   0 1 2
//   3 4 5
//   6 7 8
constexpr int kBoardCells = 9;
constexpr int8_t kEmpty = 0;
constexpr int8_t kPlayerX = 1;
constexpr int8_t kPlayerO = 2;

// The 8 index triples that count as a win: 3 rows, 3 columns, 2 diagonals.
constexpr int kWinLines[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},  // rows
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},  // columns
    {0, 4, 8}, {2, 4, 6},             // diagonals
};

// Returns kPlayerX, kPlayerO, or kEmpty (no winner yet) for `board`. Shared
// verbatim between host code (win/draw detection in the game loop) and
// device code (terminal-state checks inside the search kernels), so both
// sides agree on what counts as a win.
inline __host__ __device__ int8_t CheckWinner(const int8_t* board) {
  for (int line = 0; line < 8; ++line) {
    const int a = kWinLines[line][0];
    const int b = kWinLines[line][1];
    const int c = kWinLines[line][2];
    if (board[a] != kEmpty && board[a] == board[b] && board[b] == board[c]) {
      return board[a];
    }
  }
  return kEmpty;
}

inline __host__ __device__ bool IsBoardFull(const int8_t* board) {
  for (int i = 0; i < kBoardCells; ++i) {
    if (board[i] == kEmpty) {
      return false;
    }
  }
  return true;
}

// Writes the indices of every empty cell in `board` into `out_cells` and
// returns how many there are (at most kBoardCells).
inline __host__ __device__ int CollectEmptyCells(const int8_t* board,
                                                   int* out_cells) {
  int count = 0;
  for (int i = 0; i < kBoardCells; ++i) {
    if (board[i] == kEmpty) {
      out_cells[count++] = i;
    }
  }
  return count;
}

// Renders `board` as a 3-line grid ("X" / "O" / "." per cell) suitable for
// printing to a terminal or a run log, which is how this project satisfies
// the "show the game state to users" part of the assignment. Host-only
// (defined in src/board.cpp), unlike the functions above.
std::string FormatBoard(const int8_t* board);

#endif  // INCLUDE_BOARD_H_
