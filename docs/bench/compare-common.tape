# Shared look and setup for the llm.c comparison video (docs/bench-vs-llmc.*),
# sourced by compare-llmc.tape and compare-llmm.tape, one pane each. Don't
# render these directly: run scripts/render_compare_video.sh from a normal
# terminal on an idle machine. It first runs each trainer alone, with nothing
# else on the CPU, saving its output with timings (LOG_VIEW_RECORD); these
# tapes then film the same commands replaying those runs (LOG_VIEW_REPLAY),
# because filming loads the CPU and would slow llm.c's OpenMP threads far more
# than llm.mojo. The panes are stacked side by side and synced at the moment
# Enter is pressed.
#
# Each pane is 960x1080 with its half of build/compare/compare-bg.png as
# MarginFill (set by the pane's own tape), so the two stack into the 1920x1080
# canvas scripts/compare_frame.py draws. 25px gives ~51 columns, room for
# scripts/log_view.py's 48-column step lines.

Require make

Set Shell "bash"
Set FontFamily "IBM Plex Mono"
Set FontSize 25
Set LineHeight 1.15
Set Width 960
Set Height 1080
Set Margin 56
Set BorderRadius 16
Set Padding 14
Set WindowBar Colorful
Set CursorBlink false
Set Framerate 24
Set TypingSpeed 0ms
Set WaitTimeout 30m
Set WaitPattern />\s*$/
Set Theme { "name": "llmm-bench", "background": "#0d0f14", "foreground": "#e2e8f0", "cursor": "#ff6a32", "selection": "#2a2f3a", "black": "#0d0f14", "red": "#f87171", "green": "#4ade80", "yellow": "#facc15", "blue": "#60a5fa", "magenta": "#c084fc", "cyan": "#22d3ee", "white": "#e2e8f0", "brightBlack": "#6b7280", "brightRed": "#fca5a5", "brightGreen": "#86efac", "brightYellow": "#fde047", "brightBlue": "#93c5fd", "brightMagenta": "#d8b4fe", "brightCyan": "#67e8f9", "brightWhite": "#f3f4f6" }

# Off camera: put pixi on PATH, pin the pretty layout (an LLM harness's
# environment would otherwise switch log_view's colors off), and build the
# trainer so the recording shows training, not compiling. llm.c's binary must
# already exist: building it here would recompile it (see compare-llmc-cpu).
Hide
Type `export PATH="$(ls -d "$HOME"/.local/share/mise/installs/pixi/*.*.* | sort -V | tail -1):$PATH" LLMM_LOG=pretty PS1='\[\e[38;2;255;106;50m\]>\[\e[0m\] '`
Enter
Type "make -s build/train_gpt2 >/dev/null 2>&1; clear"
Enter
Wait
