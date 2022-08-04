from CLIP import CLIP
from torch.utils.data import DataLoader
from dataset import ImageMatchingDataset
from dataloader import load_corpus


model_name = 'ViT-B/32'
device = 'cuda'
corpus_path = 'data'


# Load corpus
corpus = ImageMatchingDataset(load_corpus(corpus_path))


if __name__ == '__main__':
    # clip_model = CLIP(model_name, device)
    # image_features = clip_model.encode_image_from_path(image_path)
    # print(image_features.shape)
    for paths in DataLoader(corpus, batch_size = 2):
        print(paths)