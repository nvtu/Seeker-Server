from clip_model import CLIP
from torch.utils.data import DataLoader
from data_helper.dataset import ImageMatchingDataset
from data_helper.dataloader import load_corpus_from_index_file
from tqdm import tqdm
import torch
import argparse
import os


# model_name = 'ViT-L/14' # Current light-weight state-of-the-art model
# device = 'cuda'
# corpus_path = '../data'
BATCH_INDEX_FOR_SAVING = 1000

parser = argparse.ArgumentParser()
parser.add_argument('corpus_path', type=str, default='../data')
parser.add_argument('output_folder_path', type=str, default='..')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--model_name', type=str, default='ViT-L/14@336px')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--starting_index', type=int, default=0)

args = parser.parse_args()


def save(features, mappings, idx):
    # Save the tensor and mapping indices
    feature_output_path = os.path.join(args.output_folder_path, f'corpus_features_{idx}.pt')
    torch.save(features, feature_output_path)

    mapping_indices_path = os.path.join(args.output_folder_path, f'corpus_feature_mapping_{idx}.pt')
    torch.save(mapping_indices, mapping_indices_path)


if __name__ == '__main__':
    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path)

    # Initialize CLIP model
    clip_model = CLIP(args.model_name, args.device)

    # Load corpus
    corpus = load_corpus_from_index_file(args.corpus_path)

    # Start processing from the given starting index as previous one was processed
    idx = args.starting_index
    if idx > 0:
        corpus = corpus[idx * args.batch_size:] # idx variable is computed by the batchsize -> Multiply by the batch_size itself to get the right index for next batch

    corpus = ImageMatchingDataset(corpus, clip_model.preprocess)

    # Encode entire corpus
    features = []
    mapping_indices = []

    for images, paths in tqdm(DataLoader(corpus, 
                batch_size = args.batch_size)):

        image_features = clip_model.encode_image(images)
        features.append(image_features.detach().cpu())
        mapping_indices += paths

        if (idx + 1) % BATCH_INDEX_FOR_SAVING == 0: # Save the features for every 1000 batches
            features = torch.cat(features, dim=0)
            
            save(features, mapping_indices, idx+1)

            # Reset features list and mapping indices list
            features = []
            mapping_indices = []


        idx += 1

    q, r = divmod(idx, BATCH_INDEX_FOR_SAVING) 
    if r > 0:
        rounded_idx = (q + 1) * BATCH_INDEX_FOR_SAVING
        features = torch.cat(features, dim=0)
        save(features, mapping_indices, rounded_idx)
