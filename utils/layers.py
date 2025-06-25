import torch 
import time
import torch.nn as nn
import torch.nn.functional as F


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, mlp_dim, num_classes, num_layers, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)


        self.pos_encoding = nn.Parameter(torch.randn((1, self.num_patches + 1, embed_dim)))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # Same Question "What's in this image?" thats why first dim is 1

        self.class_head = ClassificationHead(embed_dim, num_classes)

        self.transformer = VisionEncoder(num_layers, embed_dim, mlp_dim, num_heads, dropout)

        self.dropout = nn.Dropout(dropout)
        patch_dim = 3 * patch_size * patch_size
        self.projector = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
    
    def forward(self, x):
        x = self.get_patch_embeddings(x)
        batch_size, num_patches, _ = x.shape
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1) # Repeat cls token because we want the same Q to be asked
          
        x = torch.concat((cls_tokens, x), dim=1) # Add the cls token for each batch 
        x = x + self.pos_encoding[:, :(num_patches + 1)]

        x = self.transformer(x)

        return self.class_head(x)


    def get_patch_embeddings(self, x):
        # [batch_size, H, W, C] -> [Batch_size, num_patches, embed_dim]
        batch_size, C, H, W = x.shape
        # x = x.reshape(batch_size, H * W * C)
        # x = x.reshape(batch_size, patch_size, patch_size, C, -1)


        x = x.view(batch_size, C, H // self.patch_size, self.patch_size,  W // self.patch_size, self.patch_size)
        # print(x.shape)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        # print(x.shape)
        x = x.view(batch_size, self.num_patches, self.patch_size * self.patch_size * C)
        return self.dropout(x)


class VisionEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, mlp_dim, num_heads, dropout):
        super().__init__()
        self.encoder = self._make_layers(num_layers, embed_dim, mlp_dim, num_heads, dropout)

    def forward(self, x):
        # print(self.encoder)
        for mha, mlp in self.encoder:           
            x = x + mha(x)
            x = x + mlp(x)

        return x
    
    def _make_layers(self, num_layers, embed_dim, mlp_dim, num_heads, dropout):

        module_list = nn.ModuleList([])
        for _ in range(num_layers):
            module_list.append(
                    nn.ModuleList([
                        EfficientMultiHeadedAttention(embed_dim, num_heads, dropout),
                        MLP(embed_dim, mlp_dim, dropout)
                    ])
            )
        
        return module_list

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.classification_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x_cls = x[:, 0, :] # Extract the [cls] which contains "Whats the most important information for this patch?"
        return self.classification_head(x_cls)

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout):
        super().__init__()
        
        self.ffn = nn.Sequential(
            LayerNormalization(embed_dim),
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.ffn(x)
    
# I just found out that MultiHeadAttention is not just Attention n times
# You lied to me https://youtu.be/bX2QwpjsmuA?si=LTpbFdlAsaUMmt2h

class EfficientMultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = eps
        self.layer_norm = LayerNormalization(embed_dim)
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v= nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.projector = (
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Dropout(dropout)
            )
        )
        
    def forward(self, x):
        batch_size, num_patches, _ = x.shape
        x = self.layer_norm(x)
        
        Q = self.fc_q(x) # [batch_size, num_patches, embed_dim]
        K = self.fc_k(x) # [batch_size, num_patches, embed_dim]
        V = self.fc_v(x) # [batch_size, num_patches, embed_dim]

        Q = Q.view(batch_size, num_patches, self.num_heads, self.head_dim) # [batch_size, num_patches, num_heads, head_dim]
        K = K.view(batch_size, num_patches, self.num_heads, self.head_dim) # [batch_size, num_patches, num_heads, head_dim]
        V = V.view(batch_size, num_patches, self.num_heads, self.head_dim) # [batch_size, num_patches, num_heads, head_dim]

        Q = Q.transpose(1, 2) # [batch_size, num_heads, num_patches, head_dim]
        K = K.transpose(1, 2) # [batch_size, num_heads, num_patches, head_dim]
        V = V.transpose(1, 2) # [batch_size, num_heads, num_patches, head_dim]

        attn = Q @ K.transpose(-2, -1) # [batch_size, num_heads, num_patches, num_patches]
        scaled_attn = attn / (torch.sqrt(torch.tensor(self.head_dim)) + self.eps) 
        scaled_attn_weights = F.softmax(scaled_attn, dim=-1)
        scaled_attn_weights = self.attn_dropout(scaled_attn_weights)
        
        out_put = scaled_attn_weights @ V # [batch_size, num_heads, num_patches, head_dim]

        out = out_put.transpose(1, 2) # [batch_size, num_patches, num_heads, head_dim]

        out_concat = out.reshape(batch_size, num_patches, self.embed_dim) # # [batch_size, num_patches, num_heads * head_dim]

        return self.projector(out_concat)

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout, train=True):
        super().__init__()
        self.dropout_val = dropout
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.attn_heads = self._make_heads()
    
        if train:
            self.projector = (
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.Dropout(dropout)
                )
            )
        else:
            self.projector = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        out_attn = []
        for attn_head in self.attn_heads:
            out_attn.append(attn_head(x))
        
        out_attn = torch.cat(out_attn, dim=-1)
        
        return self.projector(out_attn)


    def _make_heads(self):
        print(self.dropout_val)
        heads = nn.ModuleList()
        for _ in range(self.num_heads):
            heads.append(Attention(self.embed_dim, self.head_dim, self.dropout_val))
        
        return heads

class Attention(nn.Module):
    def __init__(self, embed_dim, out_dim , dropout, eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.eps = eps
        self.fc_q = nn.Linear(embed_dim, out_dim)
        self.fc_k = nn.Linear(embed_dim, out_dim)
        self.fc_v= nn.Linear(embed_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Q = self.fc_q(x) # [batch_size, num_patches, out_dim]
        K = self.fc_k(x) # [batch_size, num_patches, out_dim]
        V = self.fc_v(x) # [batch_size, num_patches, out_dim]

        attn = Q @ K.transpose(-2,-1) # [batch_size, num_patches, out_dim] *  [batch_size, out_dim , num_patches] = \
        # [batch_size, num_patches, num_patches]
        scaled_attn = attn / (torch.sqrt(torch.Tensor(self.out_dim)) + self.eps)
        scaled_attn_weights = F.softmax(scaled_attn, dim=-1)
        scaled_attn_weights = self.dropout(scaled_attn_weights)
        return scaled_attn_weights @ V  # [batch_size, num_patches, num_patches] * [batch_size, num_patches, out_dim] = \
        # [batch_size, num_patches, out_dim]


class LayerNormalization(nn.Module):
    def __init__(self, embed_dim, epsilon=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = epsilon
        self.alpha = nn.Parameter(torch.ones(embed_dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(embed_dim), requires_grad=True)

    def forward(self, x): # x = [batch_size, num_patches, embed_dim]
        # dim = 0 (along rows), 1 (along columns), -1 (along last dim)
        mean = torch.mean(x, dim=-1, keepdim=True) # [batch_size, num_patches, 1]
        var = torch.var(x, dim=-1, keepdim=True, unbiased=True) # [batch_size, num_patches, 1]
        # unbaised uses N and not N - 1 for division

        x_shifted = (x-mean)
        x = x_shifted / torch.sqrt(var + self.eps)


        return x * self.alpha + self.beta
    

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    # batch_size, C, H, W = x.shape
    # patch_size = 16
    # image_size = 224
    # num_patches = (image_size // patch_size) * (image_size // patch_size)
    # x = x.view(batch_size, C, H // patch_size, patch_size,  W // patch_size, patch_size)
    # print(x.shape)
    # x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    # print(x.shape)
    # x = x.view(batch_size, num_patches, patch_size * patch_size * C)
    # print(x.shape)
    # exit()
    # x = torch.randn(2, 2, 2)
    vit_layer = ViT(image_size=224, patch_size=16, embed_dim=768, mlp_dim= 3, num_classes=2, num_layers=1, num_heads=2, dropout=0.1)
    x_layer = vit_layer(x)
    print(f"Before ViT: {x.shape}")
    print(f"After ViT: {x_layer.shape}")

    x = torch.randn(2, 2, 2) # I made below first. So, I am not changing my test case input

    
    ln = LayerNormalization(2)
    x_norm = ln(x)

    print(f"Shape before norm: {x.shape}")
    print(f"shape after norm: {x_norm.shape}")

    # x = torch.randn(2, 2, 2)
    print()

    attn = Attention(2,2, 0.1)
    x_attn = attn(x)

    print(f"Shape before attn: {x.shape}")
    print(f"shape after attn: {x_attn.shape}")
    print()

    # x = torch.randn(2, 2, 2)
    start = time.time()
    attn = MultiHeadedAttention(2, 2, 0.1)
    x_attn = attn(x)
    end = time.time()
    time_taken_01 = end-start
    print("Conceptual MultiHeadedAttention")
    print(f"Shape before attn: {x.shape}")
    print(f"Shape after attn: {x_attn.shape}")
    print(f"Time taken: {time_taken_01:5f}")
    print()
    start = time.time()
    attn = EfficientMultiHeadedAttention(2, 2, 0.1)
    x_attn = attn(x)
    end = time.time()
    time_taken_02 = end-start
    print("Efficient MultiHeadedAttention")
    print(f"Shape before attn: {x.shape}")
    print(f"Shape after attn: {x_attn.shape}")
    print(f"Time taken: {time_taken_02:5f}")

    print(f"Speed up: {time_taken_01/time_taken_02:2f}x faster") # Approx 3x faster


    ffn = MLP(2, 3, 0.2)
    x_ffn = ffn(x)
    print(f"Before MLP: {x.shape}")
    print(f"After MLP: {x_ffn.shape}")
    print()
    pred = ClassificationHead(2, 3)
    x_pred = pred(x)
    print(f"Before Classification: {x.shape}")
    print(f"After Classification: {x_pred.shape}")
