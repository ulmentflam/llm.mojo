#!/usr/bin/env bash
# Render docs/bench.gif (the README demo) from docs/bench/{verify,train}.tape.
#
# scripts/bench_gif_frame.py draws the margin the terminal sits in (captioned
# with this machine's chip) and one badge per segment. vhs records each tape
# to a raw 1x video. ffmpeg then trims the pixel-static waits (Mojo compiling,
# the model loading), plays the verify segment at 2x and the training segment
# in real time, keeps each tape's final hold untrimmed, downscales
# 1600x1200 -> 800x600 and builds a two-pass palette GIF (rectangle diffing,
# so the static gradient doesn't shimmer); gifsicle shrinks it.
#
# Run from a normal terminal on Apple Silicon (see docs/bench/common.tape).
# Needs vhs, ffmpeg, gifsicle and uv; ttyd comes from nixpkgs when missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERIFY_HOLD_S=4  # must match verify.tape's final Sleep
HOLD_S=8         # must match train.tape's final Sleep
FREEZE_MIN_S=1.5 # a static stretch at least this long is a wait...
KEEP_S=0.75      # ...of which the viewer still sees this much
OUT=build/bench
GIF="${BENCH_GIF:-docs/bench.gif}" # override for a trial render

mkdir -p "$OUT"

# Caption the frame with the machine that actually runs the demo.
chip="$(sysctl -n machdep.cpu.brand_string)"
gpu_cores="$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/ { print $2; exit }')"
caption="${chip}${gpu_cores:+ · ${gpu_cores}-core GPU} · Metal · fp32"
uv run --quiet --with pillow python -I scripts/bench_gif_frame.py "$OUT" "$caption"

record() {
	if command -v ttyd >/dev/null 2>&1; then
		vhs "$1"
	else
		nix shell nixpkgs#ttyd -c vhs "$1"
	fi
}
record docs/bench/verify.tape
record docs/bench/train.tape

duration() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }

# An ffmpeg select expression that is true inside every static stretch of
# $1 before time $2, past the first KEEP_S of it.
waits() {
	ffmpeg -hide_banner -nostats -i "$1" \
		-vf "freezedetect=n=-60dB:d=${FREEZE_MIN_S}" -an -f null - 2>&1 |
		awk -v end="$2" -v keep="$KEEP_S" '
			/freeze_start:/ { s = $NF }
			/freeze_end:/ {
				e = ($NF < end) ? $NF : end
				if (e > s + keep) { printf "%sbetween(t,%.3f,%.3f)", sep, s + keep, e; sep = "+" }
			}
			END { if (sep == "") printf "0" }'
}

fps="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$OUT/verify-raw.mp4")"
verify_end="$(awk -v d="$(duration "$OUT/verify-raw.mp4")" -v h="$VERIFY_HOLD_S" 'BEGIN { printf "%.3f", d - h }')"
train_run="$(awk -v d="$(duration "$OUT/train-raw.mp4")" -v h="$HOLD_S" 'BEGIN { printf "%.3f", d - h }')"
verify_waits="$(waits "$OUT/verify-raw.mp4" "$verify_end")"
train_waits="$(waits "$OUT/train-raw.mp4" "$train_run")"
echo "verify waits cut: $verify_waits"
echo "train waits cut:  $train_waits"

ffmpeg -loglevel error -y \
	-i "$OUT/verify-raw.mp4" -i "$OUT/train-raw.mp4" \
	-i "$OUT/badge-verify.png" -i "$OUT/badge-train.png" \
	-filter_complex "
	[0:v]select='not(${verify_waits})',setpts=N/(2*(${fps}))/TB[v0];
	[v0][2:v]overlay=0:0[verify];
	[1:v]split[ta][tb];
	[ta]trim=end=${train_run},select='not(${train_waits})',setpts=N/(${fps})/TB[t0];
	[t0][3:v]overlay=0:0[train];
	[tb]trim=start=${train_run},setpts=PTS-STARTPTS[hold];
	[verify][train][hold]concat=n=3:v=1:a=0,fps=12,scale=800:-1:flags=lanczos,split[c][d];
	[c]palettegen=max_colors=192:stats_mode=diff[p];
	[d][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle" \
	"$OUT/bench-unopt.gif"

gifsicle -O3 --lossy=30 "$OUT/bench-unopt.gif" -o "$GIF"
ls -la "$GIF"
