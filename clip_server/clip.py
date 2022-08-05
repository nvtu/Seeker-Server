import torch
import clip


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
            image_features = self.model.encode_image(images)
        return image_features