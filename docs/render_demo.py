"""Renders an animated GIF demonstrating the CUDA Multi-GPU Tic-Tac-Toe
game: the board plus each turn's per-candidate-cell kernel scores (the
exact values minimax_kernel.cu / heuristic_kernel.cu compute), used as a
visualization/demonstration asset for the Coursera submission.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

# ---- palette ----------------------------------------------------------
BG = "#0a0e15"
PANEL = "#121826"
GRID = "#2a3448"
TEXT = "#e8eef9"
TEXT_DIM = "#8d9ab0"
X_COLOR = "#52e8d6"
O_COLOR = "#ff9d5c"
GOOD = "#74e0a0"

plt.rcParams["font.family"] = "DejaVu Sans"

# ---- recorded match (X opens) -----------------------------------------
# Exact moves + per-empty-cell scores that minimax_kernel.cu (X) and
# heuristic_kernel.cu (O) compute for this board, turn by turn.
TURNS = [
    {"player": "X", "cell": 0, "candidates": {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}, "device": 0},
    {"player": "O", "cell": 4, "candidates": {1:1,2:5,3:1,4:10,5:1,6:5,7:1,8:5}, "device": 1},
    {"player": "X", "cell": 1, "candidates": {1:0,2:0,3:0,5:0,6:0,7:0,8:0}, "device": 0},
    {"player": "O", "cell": 2, "candidates": {2:50,3:1,5:1,6:5,7:1,8:5}, "device": 1},
    {"player": "X", "cell": 6, "candidates": {3:-8,5:-8,6:0,7:-8,8:-8}, "device": 0},
    {"player": "O", "cell": 3, "candidates": {3:50,5:1,7:1,8:5}, "device": 1},
    {"player": "X", "cell": 5, "candidates": {5:0,7:-8,8:-8}, "device": 0},
    {"player": "O", "cell": 8, "candidates": {7:1,8:5}, "device": 1},
    {"player": "X", "cell": 7, "candidates": {7:0}, "device": 0},
]
RESULT = "Draw"

KERNEL_NAME = {"X": "minimax_kernel.cu (exhaustive search)", "O": "heuristic_kernel.cu (1-ply scoring)"}
PLAYER_COLOR = {"X": X_COLOR, "O": O_COLOR}


def draw_board(ax, board, highlight_cell=None, highlight_alpha=1.0, win_line=None):
    ax.clear()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(PANEL)

    for i in range(4):
        ax.plot([i, i], [0, 3], color=GRID, lw=2, zorder=1)
        ax.plot([0, 3], [i, i], color=GRID, lw=2, zorder=1)

    if win_line is not None:
        for idx in win_line:
            row, col = divmod(idx, 3)
            ax.add_patch(mpatches.Rectangle((col, 2 - row), 1, 1, facecolor=GOOD, alpha=0.18, zorder=0))

    if highlight_cell is not None:
        row, col = divmod(highlight_cell, 3)
        ax.add_patch(mpatches.Rectangle(
            (col + 0.06, 2 - row + 0.06), 0.88, 0.88, fill=False,
            edgecolor=PLAYER_COLOR[board_mark_label(board[highlight_cell])],
            lw=3, alpha=highlight_alpha, zorder=3))

    for idx, v in enumerate(board):
        if v == 0:
            continue
        row, col = divmod(idx, 3)
        cx, cy = col + 0.5, 2 - row + 0.5
        if v == 1:
            m = 0.28
            ax.plot([cx - m, cx + m], [cy - m, cy + m], color=X_COLOR, lw=5, solid_capstyle="round", zorder=2)
            ax.plot([cx - m, cx + m], [cy + m, cy - m], color=X_COLOR, lw=5, solid_capstyle="round", zorder=2)
        else:
            circ = plt.Circle((cx, cy), 0.3, fill=False, edgecolor=O_COLOR, lw=5, zorder=2)
            ax.add_patch(circ)


def board_mark_label(v):
    return "X" if v == 1 else "O"


def draw_scores(ax, player, candidates, board, chosen_cell=None, reveal_count=None):
    # A lollipop plot (stem + dot) rather than bars: several of this
    # project's real minimax scores are 0 or close to it, which a bar chart
    # renders as an invisible sliver. A dot at the value is visible even at
    # height 0, so every scored candidate stays legible.
    ax.clear()
    ax.set_facecolor(PANEL)
    cells = sorted(candidates.keys())
    if reveal_count is None:
        reveal_count = len(cells)
    shown = cells[:reveal_count]
    values = [candidates[c] for c in shown]
    color = PLAYER_COLOR[player]

    xs = list(range(len(shown)))
    for x, v, c in zip(xs, values, shown):
        is_best = (c == chosen_cell)
        ax.vlines(x, 0, v, color=color, lw=3.2 if is_best else 1.8,
                   alpha=1.0 if is_best else 0.55, zorder=2)
        ax.scatter([x], [v], s=120 if is_best else 55, color=color,
                   alpha=1.0 if is_best else 0.7,
                   edgecolor=TEXT if is_best else "none", linewidth=1.6, zorder=3)
        va = "bottom" if v >= 0 else "top"
        ax.annotate(str(v), (x, v), xytext=(0, 9 if v >= 0 else -9),
                    textcoords="offset points", ha="center", va=va,
                    color=TEXT if is_best else TEXT_DIM, fontsize=9,
                    fontweight="bold" if is_best else "normal", zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels(["cell " + str(c) for c in shown], color=TEXT_DIM, fontsize=8, rotation=0)
    ax.tick_params(axis="x", length=0)
    ax.axhline(0, color=GRID, lw=1, zorder=1)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])

    lo = min(values) if values else 0
    hi = max(values) if values else 1
    pad = max((hi - lo) * 0.3, 2)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.6, max(len(cells), 1) - 0.4)

    kname = "minimax search" if player == "X" else "heuristic scoring"
    ax.set_title(
        f"Player {player} · one thread per empty cell · {kname}",
        color=color, fontsize=11, fontweight="bold", loc="left", pad=10)


def find_win_line(board):
    lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a, b, c in lines:
        if board[a] and board[a] == board[b] == board[c]:
            return (a, b, c)
    return None


# ---- build the explicit frame list ------------------------------------
# Each frame: dict describing exactly what draw_board/draw_scores should
# show. Built up-front so the animation callback just renders, never
# recomputes game state.
FPS = 10
frames = []

def add_hold(frame, n):
    for _ in range(n):
        frames.append(dict(frame))

board = [0] * 9

# Intro title card (board empty, no scores panel content)
for _ in range(int(FPS * 1.6)):
    frames.append({"kind": "intro"})

for turn_idx, turn in enumerate(TURNS):
    player = turn["player"]
    cell = turn["cell"]
    candidates = turn["candidates"]
    device = turn["device"]
    n_cells = len(candidates)

    # Phase 1: reveal candidate scores progressively (board as-is, before move)
    reveal_steps = max(n_cells, 1)
    frames_per_reveal = 2
    for r in range(1, reveal_steps + 1):
        for _ in range(frames_per_reveal):
            frames.append({
                "kind": "thinking", "board": list(board), "player": player,
                "candidates": candidates, "reveal": r, "chosen": None,
                "turn_idx": turn_idx, "device": device, "cell": cell,
            })
    # Hold on fully-revealed scores with the winner highlighted
    add_hold({
        "kind": "thinking", "board": list(board), "player": player,
        "candidates": candidates, "reveal": n_cells, "chosen": cell,
        "turn_idx": turn_idx, "device": device, "cell": cell,
    }, int(FPS * 0.9))

    # Phase 2: commit the move
    board[cell] = 1 if player == "X" else 2
    add_hold({
        "kind": "placed", "board": list(board), "player": player,
        "candidates": candidates, "chosen": cell,
        "turn_idx": turn_idx, "device": device, "cell": cell,
    }, int(FPS * 1.0))

# Outro: final result, held
win_line = find_win_line(board)
add_hold({"kind": "result", "board": list(board), "win_line": win_line}, int(FPS * 2.6))

print(f"Total frames: {len(frames)} (~{len(frames)/FPS:.1f}s at {FPS}fps)")

# ---- figure setup -------------------------------------------------------
fig = plt.figure(figsize=(10, 5.4), facecolor=BG)
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], height_ratios=[0.16, 1],
                       left=0.04, right=0.97, top=0.90, bottom=0.07, wspace=0.18, hspace=0.12)
ax_title = fig.add_subplot(gs[0, :])
ax_board = fig.add_subplot(gs[1, 0])
ax_scores = fig.add_subplot(gs[1, 1])

for ax in (ax_title,):
    ax.axis("off")

status_text = ax_title.text(0.0, 0.3, "", transform=ax_title.transAxes,
                             color=TEXT, fontsize=13, fontweight="bold", va="center")
sub_text = ax_title.text(0.0, -0.35, "", transform=ax_title.transAxes,
                          color=TEXT_DIM, fontsize=9.5, va="center")
fig.suptitle("CUDA Multi-GPU Tic-Tac-Toe — Minimax vs Heuristic Kernel",
             color=TEXT, fontsize=15, fontweight="bold", x=0.04, y=0.975, ha="left")


def render(frame):
    kind = frame["kind"]

    if kind == "intro":
        ax_board.clear(); ax_board.axis("off"); ax_board.set_facecolor(BG)
        draw_board(ax_board, [0] * 9)
        ax_scores.clear(); ax_scores.axis("off"); ax_scores.set_facecolor(BG)
        status_text.set_text("Two CUDA kernels, two GPUs, one board")
        status_text.set_color(TEXT)
        sub_text.set_text("Player X: exhaustive minimax search  ·  Player O: 1-ply heuristic scoring")
        return

    if kind == "result":
        draw_board(ax_board, frame["board"], win_line=frame.get("win_line"))
        ax_scores.clear(); ax_scores.axis("off"); ax_scores.set_facecolor(BG)
        if frame.get("win_line"):
            winner = board_mark_label(frame["board"][frame["win_line"][0]])
            status_text.set_text(f"Player {winner} wins")
            status_text.set_color(PLAYER_COLOR[winner])
        else:
            status_text.set_text("Draw — perfect play, as expected")
            status_text.set_color(GOOD)
        sub_text.set_text("Tic-tac-toe is a solved game: two strong competitors always draw.")
        return

    player = frame["player"]
    device = frame["device"]
    turn_no = frame["turn_idx"] + 1
    kernel = KERNEL_NAME[player]

    if kind == "thinking":
        reveal = frame["reveal"]
        draw_board(ax_board, frame["board"])
        draw_scores(ax_scores, player, frame["candidates"], frame["board"],
                    chosen_cell=frame["chosen"], reveal_count=reveal)
        n_cells = len(frame["candidates"])
        status_text.set_text(f"Turn {turn_no} · Player {player} · GPU {device}")
        status_text.set_color(PLAYER_COLOR[player])
        sub_text.set_text(f"Launching {kernel}: {n_cells} threads score {n_cells} candidate cells in parallel…")
        return

    if kind == "placed":
        draw_board(ax_board, frame["board"], highlight_cell=frame["cell"])
        draw_scores(ax_scores, player, frame["candidates"], frame["board"], chosen_cell=frame["chosen"])
        status_text.set_text(f"Turn {turn_no} · Player {player} · GPU {device}")
        status_text.set_color(PLAYER_COLOR[player])
        sub_text.set_text(f"Best score: cell {frame['cell']} ({frame['candidates'][frame['cell']]}) → played")
        return


anim = FuncAnimation(fig, render, frames=frames, interval=1000 / FPS)
writer = PillowWriter(fps=FPS)
out_path = "gpu_tictactoe_demo.gif"
anim.save(out_path, writer=writer)
print("Saved", out_path)
