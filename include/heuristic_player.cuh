// Copyright 2026 Celerium-Ce
#ifndef INCLUDE_HEURISTIC_PLAYER_CUH_
#define INCLUDE_HEURISTIC_PLAYER_CUH_

#include <cstdint>

// "Competitor B": picks a move for `me` by scoring every currently-empty
// cell against a fixed, non-exhaustive heuristic (win now > block an
// opponent win > take the center > take a corner > take an edge). One
// thread per empty cell computes its own cell's score in parallel, with no
// recursive search, so this kernel is deliberately cheaper and weaker than
// the minimax competitor in minimax_player.cuh — the two GPUs are running
// different strategies, not just copies of the same one. Runs on whichever
// device `device_id` names via cudaSetDevice. Writes the chosen cell index
// (0-8) to `*out_move`.
void LaunchHeuristicMove(const int8_t* h_board, int8_t me, int8_t opponent,
                          int device_id, int* out_move);

#endif  // INCLUDE_HEURISTIC_PLAYER_CUH_
