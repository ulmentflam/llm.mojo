# llm.c comparison video, left pane: llm.c's CPU trainer. See compare-common.tape.

Output build/compare/llmc-raw.mp4
Set MarginFill "build/compare/compare-bg-left.png"

Source docs/bench/compare-common.tape

# Off camera: replay the run render_compare_video.sh measured, then type the
# command. The pane opens on it and Enter lands 1 s in,
# the same instant as in compare-llmm.tape.
Type "export LOG_VIEW_REPLAY=$PWD/build/compare/llmc.timed && clear"
Enter
Wait
Type "make compare-llmc-cpu"
Show
Sleep 1s
Enter
Wait
Sleep 6s
