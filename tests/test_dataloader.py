import os
import subprocess


def test_mojo_dataloader_bridge():
    """Test the Python-to-Mojo bridge by executing the Mojo DataLoader test suite.

    The Mojo DataLoader test suite relies heavily on Python interop to generate the mock
    token binary shards (via data.utils), evaluate Python code, list files, and clean up.
    """
    # Build environment copy and inject the detected MOJO_PYTHON_LIBRARY if not present
    env = os.environ.copy()
    if "MOJO_PYTHON_LIBRARY" not in env:
        lib_dir = ".pixi/envs/default/lib"
        if os.path.exists(lib_dir):
            for filename in os.listdir(lib_dir):
                if filename.startswith("libpython3") and (
                    filename.endswith(".dylib") or filename.endswith(".so")
                ):
                    env["MOJO_PYTHON_LIBRARY"] = os.path.join(lib_dir, filename)
                    break

    # Execute the Mojo test using the same environment and include path
    cmd = ["pixi", "run", "mojo", "-I", ".", "tests/test_dataloader.mojo"]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # Output detailed information in case of failure
    if res.returncode != 0:
        print("STDOUT:")
        print(res.stdout)
        print("STDERR:")
        print(res.stderr)

    assert res.returncode == 0, (
        f"Mojo DataLoader test script crashed with exit code {res.returncode}.\n"
        f"STDERR: {res.stderr}"
    )
