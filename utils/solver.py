from utils.layers import ViT



class Solver():
    def __init__(self, image_size, patch_size, embed_dim, mlp_dim, num_classes, num_heads):

        self.model = ViT(image_size, patch_size, embed_dim, mlp_dim, num_classes, num_heads)
        