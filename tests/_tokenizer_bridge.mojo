"""Bridge: load a tokenizer .bin and print decode results for pytest."""

from std.python import Python

from llmm.tokenizer import Tokenizer


def main() raises:
    var os = Python.import_module("os")
    var path = String(os.environ["TOKENIZER_BRIDGE_PATH"])
    var ids_str = String(os.environ["TOKENIZER_BRIDGE_IDS"])
    var tokenizer = Tokenizer(path)

    print(String(tokenizer.has_initialized))
    print(String(tokenizer.vocab_size))
    print(String(tokenizer.eot_token))

    if ids_str.byte_length() == 0:
        return

    var ids = ids_str.split(",")
    for i in range(len(ids)):
        var token_id = Int(ids[i])
        var piece = tokenizer.decode(token_id)
        var piece_bytes = piece.as_bytes()
        var line = String(String(token_id) + ":")
        for b in range(len(piece_bytes)):
            line += " " + String(Int(piece_bytes[b]))
        print(line)
