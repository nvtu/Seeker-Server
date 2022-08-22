from clip_model import CLIP

model_name = 'ViT-L/14' # Current light-weight state-of-the-art model
device = 'cuda'

if __name__ == '__main__':

    text = 'A boy is in the kitchen, sitting on a chair. He gets up to look out the window.'
    model = CLIP(model_name, device)

    print(model.encode_text(text).shape)
    

