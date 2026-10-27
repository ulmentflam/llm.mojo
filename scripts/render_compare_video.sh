#!/usr/bin/env bash
# Render the llm.c comparison video: docs/bench-vs-llmc.mp4 (real time) and
# docs/bench-vs-llmc.gif (the README preview).
#
# scripts/compare_frame.py draws the 1920x1080 canvas and its two halves. Each
# trainer then runs alone with nothing else on the CPU, its scripts/log_view.py
# output saved with timings (LOG_VIEW_RECORD). Filming comes after, and runs
# nothing: vhs's headless browser loads the CPU, and llm.c's OpenMP threads lose
# far more to that than llm.mojo does, so a live take would misstate the gap.
# The tapes (docs/bench/compare-llmc.tape, compare-llmm.tape) type the real
# make commands with LOG_VIEW_REPLAY set, which prints each saved run back at
# its own pace; both press Enter 1 s in, which syncs them at launch. The faster pane then holds its last frame until the
# slower one is done, the two stack into the MP4, and the GIF plays everything
# but the final hold at COMPARE_GIF_SPEED under a badge saying so.
#
# Run from a normal terminal on an idle machine (anything else on the CPU
# slows both runs, and not equally). Needs vhs, ffmpeg (libx264), gifsicle,
# uv and llm.c's train_gpt2 built with OpenMP (its fastest CPU
# configuration); ttyd comes from nixpkgs when missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOLD_S=6 # must match the compare tapes' final Sleep
SPEED="${COMPARE_GIF_SPEED:-4}"
OUT=build/compare
MP4="${COMPARE_MP4:-docs/bench-vs-llmc.mp4}"
GIF="${COMPARE_GIF:-docs/bench-vs-llmc.gif}"
mkdir -p "$OUT"

threads="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
machine="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -m)"
uv run --quiet --with pillow python -I scripts/compare_frame.py \
	"$OUT" "$machine" "$threads" "$SPEED"

# Pass 1: measure. Nothing else may run meanwhile.
LOG_VIEW_RECORD="$ROOT/$OUT/llmc.timed" make -s compare-llmc-cpu >/dev/null
LOG_VIEW_RECORD="$ROOT/$OUT/llmm.timed" make -s compare-llmm-cpu >/dev/null
for pane in llmc llmm; do
	echo "$pane: $(tail -1 "$OUT/$pane.timed")"
done

# Pass 2: film the replays.
record() {
	if command -v ttyd >/dev/null 2>&1; then
		vhs "$1"
	else
		nix shell nixpkgs#ttyd -c vhs "$1"
	fi
}
record docs/bench/compare-llmc.tape
record docs/bench/compare-llmm.tape

duration() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
left="$(duration "$OUT/llmc-raw.mp4")"
right="$(duration "$OUT/llmm-raw.mp4")"
total="$(awk -v a="$left" -v b="$right" 'BEGIN { printf "%.3f", (a > b ? a : b) }')"
echo "llm.c pane ${left}s, llm.mojo pane ${right}s"

ffmpeg -loglevel error -y -i "$OUT/llmc-raw.mp4" -i "$OUT/llmm-raw.mp4" -filter_complex "
	[0:v]tpad=stop_mode=clone:stop_duration=$(awk -v t="$total" -v d="$left" 'BEGIN { print t - d }')[l];
	[1:v]tpad=stop_mode=clone:stop_duration=$(awk -v t="$total" -v d="$right" 'BEGIN { print t - d }')[r];
	[l][r]hstack=inputs=2,format=yuv420p[v]" \
	-map "[v]" -c:v libx264 -preset slow -crf 24 -tune animation \
	-movflags +faststart "$MP4"

race="$(awk -v t="$total" -v h="$HOLD_S" 'BEGIN { printf "%.3f", t - h }')"
ffmpeg -loglevel error -y -i "$MP4" -i "$OUT/compare-speed.png" -filter_complex "
	[0:v]split[a][b];
	[a]trim=end=${race},setpts=(PTS-STARTPTS)/${SPEED}[fast];
	[fast][1:v]overlay=0:0[race];
	[b]trim=start=${race},setpts=PTS-STARTPTS[hold];
	[race][hold]concat=n=2:v=1:a=0,fps=10,scale=960:-1:flags=lanczos,split[c][d];
	[c]palettegen=max_colors=160:stats_mode=diff[p];
	[d][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle" \
	"$OUT/compare-unopt.gif"
gifsicle -O3 --lossy=40 "$OUT/compare-unopt.gif" -o "$GIF"
ls -la "$MP4" "$GIF"
