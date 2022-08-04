import os


def load_corpus(corpus_path: str):
    corpus = [os.path.join(corpus_path, path) for path in os.listdir(corpus_path)]    
    return corpus