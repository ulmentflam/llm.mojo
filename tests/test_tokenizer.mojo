from std.testing import assert_equal, assert_true, TestSuite
from std.python import Python

from llmm.tokenizer import Tokenizer


def setup_test_files() raises:
    var sys = Python.import_module("sys")
    sys.path.append(".")

    var fixtures = Python.import_module("tests._tokenizer_fixtures")
    fixtures.write_test_tokenizer("test_tokenizer.bin")
    print("Tokenizer test file created successfully!")


def cleanup_test_files() raises:
    var os = Python.import_module("os")
    try:
        _ = os.remove("test_tokenizer.bin")
    except:
        pass


def test_tokenizer_decode() raises:
    var tokenizer = Tokenizer("test_tokenizer.bin")

    assert_true(tokenizer.has_initialized)
    assert_equal(tokenizer.vocab_size, 8)
    assert_equal(tokenizer.eot_token, 3)

    assert_equal(tokenizer.decode(0), "hello")
    assert_equal(tokenizer.decode(1), " world")
    assert_equal(tokenizer.decode(2), "!")

    var token3 = tokenizer.decode(3)
    assert_equal(token3.byte_length(), 1)
    assert_equal(Int(token3.as_bytes()[0]), 7)

    var token4 = tokenizer.decode(4)
    assert_equal(token4.byte_length(), 1)
    assert_equal(Int(token4.as_bytes()[0]), 255)

    assert_equal(tokenizer.decode(99), "")


def test_tokenizer_missing_file() raises:
    var tokenizer = Tokenizer("missing_tokenizer.bin", quiet=True)
    assert_true(not tokenizer.has_initialized)
    assert_equal(tokenizer.decode(0), "")


def main() raises:
    setup_test_files()
    try:
        TestSuite.discover_tests[__functions_in_module()]().run()
    except e:
        cleanup_test_files()
        raise e^
    cleanup_test_files()
