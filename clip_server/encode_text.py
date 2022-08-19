from clip_model import CLIP

model_name = 'ViT-L/14' # Current light-weight state-of-the-art model
device = 'cuda'

if __name__ == '__main__':

    text = 'This is a test'
    model = CLIP(model_name, device)

    print(model.encode_text(text).shape)
    

