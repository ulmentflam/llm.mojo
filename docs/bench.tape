# Records the README demo (docs/bench.gif): `make verify` on the CPU and Metal
# GPU correctness gates, then llm.c's canonical 40-step Tiny Shakespeare run
# ending on a generated sample. Apple Silicon only: the throughput it shows is
# the M4 Max's.
#
# Don't render this file directly; scripts/render_bench_gif.sh renders it to
# a raw 1x video, trims the static stretches while Mojo compiles, speeds the
# run up 2x behind a badge saying both, and keeps the final 6 s hold (the
# tape's last Sleep) at 1x. Run that script from a normal
# terminal: a background-priority shell (an agent's sandbox, say) runs the
# trainer about 2x slower and the GIF would show the wrong tok/s.

Output build/bench/bench-raw.mp4

Require make

Set Shell "bash"
# Geometry shared with scripts/bench_gif_frame.py, which draws the margin
# (glow, window shadow, caption) into build/bench/bench-bg.png.
Set FontFamily "IBM Plex Mono"
# 19px leaves ~124 columns, enough for a training step line without wrapping.
Set FontSize 19
Set Width 1600
Set Height 900
Set Margin 72
Set MarginFill "build/bench/bench-bg.png"
Set BorderRadius 16
Set Padding 20
Set WindowBar Colorful
# A steady cursor keeps compile waits pixel-static, so the render script can
# find and trim them.
Set CursorBlink false
Set Framerate 24
Set TypingSpeed 60ms
Set WaitTimeout 20m
Set WaitPattern />\s*$/
Set Theme { "name": "llmm-bench", "background": "#0d0f14", "foreground": "#e2e8f0", "cursor": "#ff6a32", "selection": "#2a2f3a", "black": "#0d0f14", "red": "#f87171", "green": "#4ade80", "yellow": "#facc15", "blue": "#60a5fa", "magenta": "#c084fc", "cyan": "#22d3ee", "white": "#d1d5db", "brightBlack": "#6b7280", "brightRed": "#fca5a5", "brightGreen": "#86efac", "brightYellow": "#fde047", "brightBlue": "#93c5fd", "brightMagenta": "#d8b4fe", "brightCyan": "#67e8f9", "brightWhite": "#f3f4f6" }

# vhs starts bash with a minimal environment, so put the newest mise-managed
# pixi on PATH and build the trainer before recording; the demo then shows
# verification and training, not compilation.
Hide
Type `export PATH="$(ls -d "$HOME"/.local/share/mise/installs/pixi/*.*.* | sort -V | tail -1):$PATH" PS1='\[\e[38;2;255;106;50m\]>\[\e[0m\] '`
Enter
Type "make -s build/train_gpt2 && clear"
Enter
Wait
Show

Type "make verify"
Sleep 500ms
Enter
Wait
Sleep 2s

Hide
Type "clear"
Enter
Wait
Show

Type "make train ARGS='-x 40 -b 4 -t 64 -l 1e-4 -m 5 -s 40'"
Sleep 500ms
Enter
Wait

# Hold on the sample. render_bench_gif.sh keeps exactly this tail at 1x.
Sleep 6s
