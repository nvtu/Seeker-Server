import os


def load_corpus_from_folder(corpus_path: str):
    corpus = sorted([os.path.join(corpus_path, path) for path in os.listdir(corpus_path)])
    return corpus


def load_corpus_from_index_file(index_file_path: str):
    """
    Load corpus from an index file containing the paths of the images.
    """
    with open(index_file_path, 'r') as f:
        corpus = sorted([line.strip() for line in f])
    return corpus