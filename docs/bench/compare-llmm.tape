# llm.c comparison video, right pane: llm.mojo on CPU. See compare-common.tape.

Output build/compare/llmm-raw.mp4
Set MarginFill "build/compare/compare-bg-right.png"

Source docs/bench/compare-common.tape

# Off camera: replay the run render_compare_video.sh measured, then type the
# command. The pane opens on it and Enter lands 1 s in,
# the same instant as in compare-llmc.tape.
Type "export LOG_VIEW_REPLAY=$PWD/build/compare/llmm.timed && clear"
Enter
Wait
Type "make compare-llmm-cpu"
Show
Sleep 1s
Enter
Wait
Sleep 6s
