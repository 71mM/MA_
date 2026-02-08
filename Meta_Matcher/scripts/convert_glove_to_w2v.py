import os


def convert_glove_to_word2vec(glove_input_path: str, w2v_output_path: str):
    """
    Converts a GloVe text file (no header) to Word2Vec text format (with header).
    """
    os.makedirs(os.path.dirname(w2v_output_path), exist_ok=True)

    vocab_size = 0
    vector_dim = None

    # First pass: count vocab + dim
    with open(glove_input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 10:
                continue
            if vector_dim is None:
                vector_dim = len(parts) - 1
            vocab_size += 1

    if vector_dim is None:
        raise ValueError("Could not infer embedding dimension from GloVe file.")

    # Second pass: write Word2Vec file
    with open(w2v_output_path, "w", encoding="utf-8") as out:
        out.write(f"{vocab_size} {vector_dim}\n")
        with open(glove_input_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                out.write(line)

    print("   Vocab size:", vocab_size)
    print("   Vector dim:", vector_dim)
    print("   Output:", w2v_output_path)


if __name__ == "__main__":
    glove_input = r"../embedders/embeddings/glove/glove.840B.300d.txt"
    w2v_output  = r"../embedders/embeddings/glove/glove.840B.300d.w2v.txt"

    convert_glove_to_word2vec(glove_input, w2v_output)
