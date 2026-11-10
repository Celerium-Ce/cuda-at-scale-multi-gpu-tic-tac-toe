// Copyright 2026 Celerium-Ce
//
// CUDA Multi-GPU Tic-Tac-Toe.
//
// Drives a full game of tic-tac-toe between two GPU "competitors": Player
// X decides every move with an exhaustive minimax search kernel
// (minimax_kernel.cu), and Player O decides every move with a shallower
// heuristic-scoring kernel (heuristic_kernel.cu). If the machine has 2 or
// more CUDA devices, each player's kernels run on its own physical GPU
// (cudaSetDevice per move); on a single-GPU machine (such as the Coursera
// lab), both competitors fall back to sharing device 0, which this driver
// reports clearly instead of silently pretending otherwise.

#include <sys/stat.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "include/board.h"
#include "include/cuda_utils.h"
#include "include/heuristic_player.cuh"
#include "include/minimax_player.cuh"

namespace {

struct Args {
  std::string log_file = "logs/game.log";
  int8_t first_player = kPlayerX;
};

void PrintUsage(const char* prog_name) {
  std::cout
      << "Usage: " << prog_name << " [options]\n"
      << "  --log FILE     Path to write the game log (default: logs/game.log)\n"
      << "  --first X|O    Which competitor moves first (default: X)\n"
      << "  -h, --help     Show this message\n";
}

bool ParseArgs(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto NextValue = [&](const char* flag_name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << flag_name << "\n";
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };

    if (arg == "--log") {
      args->log_file = NextValue("--log");
    } else if (arg == "--first") {
      const std::string value = NextValue("--first");
      if (value == "X" || value == "x") {
        args->first_player = kPlayerX;
      } else if (value == "O" || value == "o") {
        args->first_player = kPlayerO;
      } else {
        std::cerr << "--first must be X or O\n";
        return false;
      }
    } else if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage(argv[0]);
      return false;
    }
  }
  return true;
}

std::string ParentDir(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  return (slash == std::string::npos) ? std::string() : path.substr(0, slash);
}

bool DirExists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

// Creates `path` if it does not already exist. Only handles a single path
// component (the log file's immediate parent), unlike the sibling project's
// recursive MakeDirsRecursive, since the game log never nests more than one
// directory deep.
void EnsureDirExists(const std::string& path) {
  if (!path.empty() && !DirExists(path)) {
    mkdir(path.c_str(), 0755);
  }
}

const char* PlayerName(int8_t player) {
  return (player == kPlayerX) ? "X (minimax GPU)" : "O (heuristic GPU)";
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    return EXIT_FAILURE;
  }

  EnsureDirExists(ParentDir(args.log_file));

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    std::cerr << "No CUDA-capable device found.\n";
    return EXIT_FAILURE;
  }

  const int device_x = 0;
  const int device_o = (device_count >= 2) ? 1 : 0;

  std::ofstream log(args.log_file);
  auto LogLine = [&](const std::string& line) {
    std::cout << line << "\n";
    log << line << "\n";
  };

  LogLine("CUDA Multi-GPU Tic-Tac-Toe");
  LogLine("Devices visible: " + std::to_string(device_count));
  for (int dev = 0; dev < device_count; ++dev) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    LogLine("  [" + std::to_string(dev) + "] " + prop.name);
  }
  if (device_count >= 2) {
    LogLine("Player X runs on device 0, Player O runs on device 1.");
  } else {
    LogLine("Only one GPU is visible, so both players share device 0 "
            "(this still runs each player's own kernel per move; it just "
            "can't put them on separate physical GPUs here).");
  }
  LogLine("");

  int8_t board[kBoardCells] = {kEmpty, kEmpty, kEmpty, kEmpty, kEmpty,
                                kEmpty, kEmpty, kEmpty, kEmpty};
  int8_t current_player = args.first_player;
  int turn = 1;

  while (true) {
    const int8_t winner = CheckWinner(board);
    if (winner != kEmpty) {
      LogLine(std::string(PlayerName(winner)) + " wins on turn " +
              std::to_string(turn - 1) + "!");
      break;
    }
    if (IsBoardFull(board)) {
      LogLine("Draw: the board filled up with no winner.");
      break;
    }

    const auto move_start = std::chrono::high_resolution_clock::now();
    int move = -1;
    if (current_player == kPlayerX) {
      LaunchMinimaxMove(board, kPlayerX, kPlayerO, device_x, &move);
    } else {
      LaunchHeuristicMove(board, kPlayerO, kPlayerX, device_o, &move);
    }
    const auto move_end = std::chrono::high_resolution_clock::now();
    const double move_ms =
        std::chrono::duration<double, std::milli>(move_end - move_start)
            .count();

    board[move] = current_player;

    LogLine("Turn " + std::to_string(turn) + ": " +
            PlayerName(current_player) + " plays cell " +
            std::to_string(move) + " (" + std::to_string(move_ms) +
            " ms on device " +
            std::to_string(current_player == kPlayerX ? device_x : device_o) +
            ")");
    LogLine(FormatBoard(board));

    current_player = (current_player == kPlayerX) ? kPlayerO : kPlayerX;
    ++turn;
  }

  return EXIT_SUCCESS;
}
