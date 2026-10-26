#!/usr/bin/env bash
# Render docs/bench.gif (the README demo) from docs/bench.tape.
#
# scripts/bench_gif_frame.py draws the margin the terminal sits in (captioned
# with this machine's chip) and the speed badge. vhs records the tape to a raw
# 1x video. ffmpeg then trims the static compile waits from everything but the
# final hold, plays that at 2x under the badge, keeps the hold at 1x,
# downscales 1600x900 -> 800x450 and builds a
# two-pass palette GIF (rectangle diffing, so the static gradient doesn't
# shimmer); gifsicle shrinks it.
#
# Run from a normal terminal on Apple Silicon (see docs/bench.tape's header).
# Needs vhs, ffmpeg, gifsicle and uv; ttyd comes from nixpkgs when missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOLD_S=6 # must match the tape's final Sleep
OUT=build/bench
RAW="$OUT/bench-raw.mp4"
BADGE="$OUT/badge-2x.png"
GIF="${BENCH_GIF:-docs/bench.gif}" # override for a trial render

mkdir -p "$OUT"

# Caption the frame with the machine that actually runs the demo.
chip="$(sysctl -n machdep.cpu.brand_string)"
gpu_cores="$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/ { print $2; exit }')"
caption="GPT-2 124M · fp32 · ${chip}${gpu_cores:+, ${gpu_cores}-core GPU} · Metal"
uv run --quiet --with pillow python -I scripts/bench_gif_frame.py "$OUT" "$caption"

if command -v ttyd >/dev/null 2>&1; then
	vhs docs/bench.tape
else
	nix shell nixpkgs#ttyd -c vhs docs/bench.tape
fi

duration="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$RAW")"
fps="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$RAW")"
run_s="$(awk -v d="$duration" -v h="$HOLD_S" 'BEGIN { printf "%.3f", d - h }')"

# Compile waits leave the screen pixel-static. Find every static stretch of
# FREEZE_MIN_S or more before the hold, keep its first KEEP_S so the viewer
# sees the pause, and drop the rest. The badge says the waits were cut.
FREEZE_MIN_S=1.5
KEEP_S=0.75
drop="$(ffmpeg -hide_banner -nostats -i "$RAW" \
	-vf "freezedetect=n=-60dB:d=${FREEZE_MIN_S}" -an -f null - 2>&1 |
	awk -v run="$run_s" -v keep="$KEEP_S" '
		/freeze_start:/ { s = $NF }
		/freeze_end:/ {
			e = ($NF < run) ? $NF : run
			if (e > s + keep) { printf "%sbetween(t,%.3f,%.3f)", sep, s + keep, e; sep = "+" }
		}
		END { if (sep == "") printf "0" }')"
echo "dropping static stretches: $drop"

ffmpeg -loglevel error -y -i "$RAW" -i "$BADGE" -filter_complex "
	[0:v]split[a][b];
	[a]trim=end=${run_s},select='not(${drop})',setpts=N/(2*(${fps}))/TB[fast];
	[fast][1:v]overlay=0:0[run];
	[b]trim=start=${run_s},setpts=PTS-STARTPTS[hold];
	[run][hold]concat=n=2:v=1:a=0,fps=12,scale=800:-1:flags=lanczos,split[c][d];
	[c]palettegen=max_colors=192:stats_mode=diff[p];
	[d][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle" \
	"$OUT/bench-unopt.gif"

gifsicle -O3 --lossy=30 "$OUT/bench-unopt.gif" -o "$GIF"
ls -la "$GIF"
