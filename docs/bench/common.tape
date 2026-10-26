# Shared look and setup for the README demo (docs/bench.gif), sourced by
# verify.tape and train.tape. Don't render these directly: run
# scripts/render_bench_gif.sh from a normal terminal on Apple Silicon (a
# background-priority shell, an agent's sandbox say, runs the trainer slower
# and the GIF would show the wrong tok/s).
#
# Readability drives the geometry. The trainer and `make verify` print their
# compact layout (llmm/term.mojo), every line under ~75 columns, so the font
# can be large: 28px in a 1600x1200 frame is ~81 columns (the train command
# line is 79) by ~30 rows, which the 800px README render shows at 14px.

Require make

Set Shell "bash"
# Geometry shared with scripts/bench_gif_frame.py, which draws the margin
# (glow, window shadow, caption) into build/bench/bench-bg.png.
Set FontFamily "IBM Plex Mono"
Set FontSize 28
Set LineHeight 1.15
Set Width 1600
Set Height 1200
Set Margin 72
Set MarginFill "build/bench/bench-bg.png"
Set BorderRadius 16
Set Padding 16
Set WindowBar Colorful
# A steady cursor keeps waits pixel-static, so the render script can find and
# trim them.
Set CursorBlink false
Set Framerate 24
Set TypingSpeed 45ms
Set WaitTimeout 20m
Set WaitPattern />\s*$/
Set Theme { "name": "llmm-bench", "background": "#0d0f14", "foreground": "#e2e8f0", "cursor": "#ff6a32", "selection": "#2a2f3a", "black": "#0d0f14", "red": "#f87171", "green": "#4ade80", "yellow": "#facc15", "blue": "#60a5fa", "magenta": "#c084fc", "cyan": "#22d3ee", "white": "#e2e8f0", "brightBlack": "#6b7280", "brightRed": "#fca5a5", "brightGreen": "#86efac", "brightYellow": "#fde047", "brightBlue": "#93c5fd", "brightMagenta": "#d8b4fe", "brightCyan": "#67e8f9", "brightWhite": "#f3f4f6" }

# vhs starts bash with a minimal environment, so put the newest mise-managed
# pixi on PATH and build the trainer before recording; the demo then shows
# verification and training, not compilation. The recorded shell inherits the
# caller's environment, so pin the pretty format at the normal level: rendered
# from an LLM harness, the output would otherwise switch to the llm format.
Hide
Type `export PATH="$(ls -d "$HOME"/.local/share/mise/installs/pixi/*.*.* | sort -V | tail -1):$PATH" LLMM_LOG=pretty LLMM_LOG_LEVEL=normal LLMM_LOG_TIME= LLMM_LOG_FILE= PS1='\[\e[38;2;255;106;50m\]>\[\e[0m\] '`
Enter
Type "make -s build/train_gpt2 && clear"
Enter
Wait
Show
