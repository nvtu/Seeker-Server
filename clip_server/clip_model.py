from locale import normalize
import torch
import clip
from sklearn.preprocessing import normalize
import numpy as np


class CLIP:

    
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        self.device = device

        # Initialize model
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)

    
    def encode_image(self, images):
        """
        Encodes an image from a tensor using CLIP model.
        """
        with torch.no_grad():
            # images = self.preprocess(images).to(self.device)
            image_features = self.model.encode_image(images)
        return image_features


    def encode_text(self, text):
        """
        Encodes text into a vector using CLIP model.
        NOTE: Encode only one sentence.
        """
        with torch.no_grad():
            tokenized_text = clip.tokenize(text).to(self.device)
            text_features = self.model.encode_text(tokenized_text)
            text_features = torch.Tensor(normalize(text_features.detach().cpu().reshape(1, -1)).flatten().tolist())
        return text_features