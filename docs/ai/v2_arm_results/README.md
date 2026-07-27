# v2 precision re-run results

One JSON fragment per arm, written by `scripts/record_arm_result.py` as that
arm finishes training and is scored. `scripts/update_readme_results.py`
regenerates the README's precision table from whichever fragments exist.

These are generated files. Every number is parsed from the run's own
`train.log` and `make eval` output — nothing is entered by hand, so the table
cannot claim something the runs did not measure. The previous table was
maintained by hand and stood for days asserting a precision comparison its
runs did not support.
