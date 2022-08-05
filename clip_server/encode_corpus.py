from clip import CLIP
from torch.utils.data import DataLoader
from data_helper.dataset import ImageMatchingDataset
from data_helper.dataloader import load_corpus
from tqdm import tqdm
import torch


model_name = 'ViT-B/32'
device = 'cuda'
corpus_path = '../data'


if __name__ == '__main__':
    # Initialize CLIP model
    clip_model = CLIP(model_name, device)

    # Load corpus
    corpus = ImageMatchingDataset(load_corpus(corpus_path), clip_model.preprocess)

    # Encode entire corpus
    features = []
    mapping_indices = []
    for images, paths in tqdm(DataLoader(corpus, batch_size = 2)):
        image_features = clip_model.encode_image(images)
        features.append(image_features)
        mapping_indices += paths

    # Convert into tensor
    features = torch.cat(features, dim=0)

    # Save the tensor and mapping indices
    feature_path = '../data_corpus.pt'
    torch.save(features, feature_path)

    mapping_indices_path = '../data_corpus_mapping_indices.pt'
    torch.save(mapping_indices, mapping_indices_path)