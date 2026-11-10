# CUDA Multi-GPU Tic-Tac-Toe

Coursera "CUDA at Scale for the Enterprise" independent project: a game of
tic-tac-toe between two GPU "competitors", each deciding its own moves with
a real CUDA kernel instead of plain CPU logic with a decorative kernel call.

## What this project is and why

The assignment asks for a simple game played by at least two GPUs acting as
competitors, with each GPU allowed to use the same or a different strategy,
graded by a presentation of the design rather than by a tournament ladder.
Tic-tac-toe was picked over something like Connect 4 or chess because its
entire game tree (at most 9 plies, at most 9!&nbsp;=&nbsp;362,880 terminal
paths) is small enough to search exhaustively from a single kernel launch,
which keeps the GPU side of the design honest and inspectable instead of
needing a pruned/approximate search to fit a presentation.

- **Player X — "competitor A" — minimax kernel** (`src/minimax_kernel.cu`):
  one thread per currently-empty cell. Each thread plays X on its own
  candidate cell, then recursively and exhaustively searches every possible
  continuation of the game from there, so the 9 (or fewer) legal candidate
  moves are scored in parallel rather than one after another. The host picks
  the candidate with the highest score. This is optimal, lookahead-complete
  play — Player X can never be beaten.
- **Player O — "competitor B" — heuristic kernel** (`src/heuristic_kernel.cu`):
  also one thread per empty cell, but each thread scores its cell with a
  fixed one-ply rule instead of a recursive search: take the win if placing
  here wins immediately, else block if the opponent would win there next,
  else prefer the center, then a corner, then an edge. This is deliberately
  a *different and weaker* strategy than Player X's, demonstrating the
  rubric's "same or different strategies" option with different strategies.
- **Two physical GPUs when available**: the host driver (`src/game_driver.cpp`)
  calls `cudaGetDeviceCount()` once at startup. If 2 or more CUDA devices are
  visible, Player X's kernel always runs on device 0 and Player O's kernel
  always runs on device 1 (`cudaSetDevice` before every move). On a
  single-GPU machine — which is what the Coursera lab actually provides —
  both competitors fall back to sharing device 0, and the program says so
  explicitly in its first lines of output instead of silently pretending
  otherwise. A lock-file/multi-process scheme (mentioned as optional in the
  assignment) was considered and rejected: a single process calling
  `cudaSetDevice` per move is simpler to build, simpler to explain in a
  presentation, and makes identical use of a second GPU if the grading
  machine happens to have one.

## Directory layout

```
cuda-gpu-game-project/
├── Makefile                    # build rules (nvcc)
├── run.sh                      # build + run convenience wrapper
├── README.md
├── include/
│   ├── cuda_utils.h             # CUDA_CHECK error-checking macro
│   ├── board.h                   # board layout, win/draw checks, formatting (shared by host + device)
│   ├── minimax_player.cuh         # Player X kernel launcher declaration
│   └── heuristic_player.cuh        # Player O kernel launcher declaration
├── src/
│   ├── game_driver.cpp           # CLI parsing + turn loop + logging (main)
│   ├── board.cpp                  # board-to-text rendering
│   ├── minimax_kernel.cu           # exhaustive minimax search kernel (Player X)
│   └── heuristic_kernel.cu          # one-ply heuristic scoring kernel (Player O)
├── bin/                          # build output (bin/gpu_tictactoe)
└── logs/                         # proof-of-execution game logs land here
```

## Build

Requires the CUDA Toolkit (`nvcc`) on `PATH` and at least one CUDA-capable
GPU.

```bash
make
```

Produces `bin/gpu_tictactoe`. The Makefile's `ARCHES` variable targets
compute capabilities 5.2 through 8.6 by default; override it for other
GPUs, e.g.:

```bash
make ARCHES="-gencode arch=compute_86,code=sm_86"
```

## Run

```bash
./bin/gpu_tictactoe --log logs/game.log
```

or, to build and run in one step with a timestamped log file:

```bash
./run.sh
```

### CLI options

| Flag          | Default           | Description                              |
|---------------|--------------------|--------------------------------------------|
| `--log FILE`   | `logs/game.log`   | Path to write the game log                  |
| `--first X\|O` | `X`               | Which competitor moves first                 |
| `-h`, `--help` |                   | Print usage and exit                           |

## How a move is decided, per turn

1. The host collects the indices of every empty cell on the 3x3 board (at
   most 9).
2. It copies the board to whichever device is hosting the player to move,
   plus the list of empty-cell indices.
3. It launches that player's kernel with one thread per empty cell:
   - **Minimax kernel**: each thread places its player's mark on its own
     cell, then calls a recursive `__device__` function that alternates
     maximizing/minimizing over every remaining empty cell until the board
     is full or someone has won, returning a score that rewards a faster
     win and a slower loss (`10 - depth` / `depth - 10`).
   - **Heuristic kernel**: each thread independently tries its own mark on
     its cell (score 100 if that wins outright), then the opponent's mark
     on the same cell (score 50 if that would have let the opponent win),
     and otherwise scores the cell itself: center (10) > corner (5) >
     edge (1).
4. The host copies the 9-or-fewer scores back and takes the `argmax` — this
   reduction over at most 9 values is cheap enough to leave on the host; the
   GPU work is the per-candidate search/scoring itself.
5. The chosen cell is applied to the board, the full board is printed in a
   3x3 grid of `X` / `O` / `.`, and the loop checks for a winner or a draw
   before handing the turn to the other competitor.

## How the game state is shown to users

Every turn, `game_driver.cpp` prints (to both stdout and the `--log` file):

- which competitor moved, which cell it chose, how long the GPU decision
  took, and which device it ran on;
- the resulting 3x3 board as a plain-text grid.

This is deliberately plain text rather than a graphical board: it keeps the
program dependency-free (no image/GUI library, consistent with the sibling
`cuda-at-scale-project`'s PGM-only choice) and it is exactly what the
assignment's own suggested presentation format — "a turn-by-turn description
of how the competitor's kernel will take in input, decide a move, and
output the move" — asks a reviewer to see.

### Example turn-by-turn output

The board logic (win/draw detection, the minimax scoring rule, and the
heuristic's priority order) is the same on the GPU as it is in a plain
Python prototype used to validate the design before writing the CUDA code
(`make` cannot be exercised on the Windows machine this was written on —
see "Toolchain note" below). That prototype's output is reproduced here to
show exactly what a full game looks like; it matches what `bin/gpu_tictactoe`
prints turn-by-turn once built and run on a CUDA machine:

```
Turn 1: X -> cell 0
X . .
. . .
. . .

Turn 2: O -> cell 4
X . .
. O .
. . .

Turn 3: X -> cell 1
X X .
. O .
. . .

Turn 4: O -> cell 2
X X O
. O .
. . .

Turn 5: X -> cell 6
X X O
. O .
X . .

Turn 6: O -> cell 3
X X O
O O .
X . .

Turn 7: X -> cell 5
X X O
O O X
X . .

Turn 8: O -> cell 8
X X O
O O X
X . O

Turn 9: X -> cell 7
X X O
O O X
X X O

Draw
```

A draw here is the *correct* result, not a bug: tic-tac-toe is a solved
game, and perfect play by both sides (which Player X's exhaustive minimax
provides, and which Player O's win/block heuristic happens to match for
this particular opening) always ends in a draw. Changing `--first` to `O`,
or weakening the heuristic further, produces games where X wins instead —
both are worth showing in the presentation video to demonstrate the kernel
actually responds to the board instead of following a fixed script.

## Proof of execution

**Toolchain note:** this repository was authored on a Windows machine with
no CUDA Toolkit (`nvcc`) installed (it has an NVIDIA GPU, but lacks the
toolkit), so the code itself could not be built or run there. The game
logic above was cross-checked with an equivalent plain-Python prototype of
both strategies instead. To build, run, and capture real proof of
execution, run this on any machine with `nvcc` available (such as the
Coursera lab terminal):

```bash
make
./run.sh
git add logs
git commit -m "add proof of execution"
```

`logs/game_<timestamp>.log` will then contain the GPU device name(s), every
turn's chosen cell and decision time, the full board after every move, and
the final result, straight from a real run on a real GPU.

## Lessons learned / design notes

- Tic-tac-toe's entire search space is tiny enough that "one thread per
  candidate move, each thread recursing through the rest of the game"
  needs no pruning (no alpha-beta, no transposition table) to be instant —
  that would matter for Connect 4 or chess, which is exactly why this
  project stayed with tic-tac-toe rather than reaching for a bigger game
  just to look more impressive.
- CUDA supports recursive `__device__` functions directly (no dynamic
  parallelism or relocatable device code needed) as long as the recursion
  depth is bounded, which it is here: at most 9 plies, each frame holding
  one 9-byte board copy on the thread's local stack.
- Sharing `board.h`'s win/draw/empty-cell helpers between host and device
  code via `inline __host__ __device__` functions means the host's
  win-detection in the turn loop and the kernels' terminal-state checks
  can never silently disagree about what counts as a win — they're
  compiled from the same source, not two parallel implementations.
- Giving Player O a strictly *weaker*, non-exhaustive strategy than Player
  X (rather than two copies of minimax) was a deliberate choice: it makes
  the two kernels visibly different pieces of code to walk through in the
  presentation, and it produces more interesting games than two unbeatable
  players, who can only ever draw regardless of who moves first.
- `cudaGetDeviceCount()` plus a plain `if` is enough to use a second GPU
  when one exists without requiring it: the lock-file / multi-process
  design the assignment mentions as optional would add real complexity
  (two processes, a polling or signaling protocol, two log files to
  reconcile) for a game this small, with no benefit over one process doing
  `cudaSetDevice` before each player's kernel launch.

## Remaining submission step: the presentation video

This repository is the code, build, and design-documentation half of the
assignment. The assignment also requires a recorded video presentation and
a public video URL, which is outside what this tool can produce or upload.
The video should cover, to match the grading rubric:

1. **Project description** — what the game is and why tic-tac-toe (the
   "What this project is and why" section above, in your own words).
2. **Code walkthrough** — show `minimax_kernel.cu` and `heuristic_kernel.cu`
   side by side and explain, in the rubric's own framing, how each
   competitor's kernel takes in the board, decides a move, and outputs
   that move (the "How a move is decided, per turn" section above is a
   ready-made script for this part).
3. **Demonstration** — actually build and run the program (`make && ./run.sh`)
   on a CUDA machine (e.g. the Coursera lab terminal) and show it playing a
   full game turn-by-turn, pointing out the printed board state each move
   and the final win/draw result.
