// Copyright 2026 Celerium-Ce
#include "include/board.h"

#include <cstdint>
#include <string>

namespace {

char CellSymbol(int8_t cell) {
  if (cell == kPlayerX) {
    return 'X';
  }
  if (cell == kPlayerO) {
    return 'O';
  }
  return '.';
}

}  // namespace

std::string FormatBoard(const int8_t* board) {
  std::string out;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      out += CellSymbol(board[row * 3 + col]);
      if (col < 2) {
        out += ' ';
      }
    }
    out += '\n';
  }
  return out;
}
