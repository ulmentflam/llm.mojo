# README demo, part 2: llm.c's canonical 40-step Tiny Shakespeare run, ending
# on a generated sample. Rendered by scripts/render_bench_gif.sh in real time,
# with the static model-load wait trimmed and the final hold kept. See
# common.tape.

Output build/bench/train-raw.mp4

Source docs/bench/common.tape

Type "make train ARGS='-x 40 -b 4 -t 64 -l 1e-4 -m 5 -s 40'"
Sleep 500ms
Enter
Wait

# Hold on the sample. render_bench_gif.sh keeps exactly this tail untrimmed.
Sleep 8s
