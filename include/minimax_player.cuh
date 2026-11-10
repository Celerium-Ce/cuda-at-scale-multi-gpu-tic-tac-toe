// Copyright 2026 Celerium-Ce
#ifndef INCLUDE_MINIMAX_PLAYER_CUH_
#define INCLUDE_MINIMAX_PLAYER_CUH_

#include <cstdint>

// "Competitor A": picks a move for `me` (kPlayerX or kPlayerO) by running an
// exhaustive minimax game-tree search on GPU `device_id`. One thread is
// launched per currently-empty cell; each thread places `me` on its own
// candidate cell and recursively searches the rest of the game to a
// terminal state, so every candidate move is scored in parallel instead of
// one after another. Runs on whichever device `device_id` names via
// cudaSetDevice, so the two competitors can live on two different physical
// GPUs. Writes the chosen cell index (0-8) to `*out_move`.
void LaunchMinimaxMove(const int8_t* h_board, int8_t me, int8_t opponent,
                        int device_id, int* out_move);

#endif  // INCLUDE_MINIMAX_PLAYER_CUH_
