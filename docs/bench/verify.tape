# README demo, part 1: the CPU and Metal GPU correctness gates. Rendered by
# scripts/render_bench_gif.sh, which trims the static compile waits and plays
# this part at 2x. See common.tape.

Output build/bench/verify-raw.mp4

Source docs/bench/common.tape

Type "make verify"
Sleep 500ms
Enter
Wait

# Let both passed gates register. render_bench_gif.sh never trims this tail.
Sleep 4s
