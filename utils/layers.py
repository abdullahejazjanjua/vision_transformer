import torch 
import time
import torch.nn as nn
import torch.nn.functional as F

# TODO: Implement PatchEmbedding
# TODO: Implement PositionalEncoding
# TODO: Implement MLP
# TODO: Implement TransformerBlock
# TODO: Implement ClassificationHead

class ViT(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_encoding = PositionalEncoding()
        self.layer_norm_01 = LayerNormalization(embed_dim)
        self.MHA = EfficientMultiHeadedAttention(embed_dim, num_classes)
        self.layer_norm_01 = LayerNormalization(embed_dim)
        self.mlp = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        pass

# I just found out that MultiHeadAttention is not just Attention n times
# You lied to me https://youtu.be/bX2QwpjsmuA?si=LTpbFdlAsaUMmt2h

class EfficientMultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.eps = eps
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v= nn.Linear(embed_dim, embed_dim)

        self.projector = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, num_patches, _ = x.shape
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
        scaled_attn = attn / (torch.sqrt(torch.Tensor(self.head_dim)) + self.eps) 
        scaled_attn_weights = F.softmax(scaled_attn, dim=-1)
        
        out_put = scaled_attn_weights @ V # [batch_size, num_heads, num_patches, head_dim]

        out = out_put.transpose(1, 2) # [batch_size, num_patches, num_heads, head_dim]

        out_concat = out.reshape(batch_size, num_patches, self.embed_dim)

        return self.projector(out_concat)

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.attn_heads = self._make_heads()
        self.projector = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        out_attn = []
        for attn_head in self.attn_heads:
            out_attn.append(attn_head(x))
        
        out_attn = torch.cat(out_attn, dim=-1)
        
        return self.projector(out_attn)


    def _make_heads(self):
        heads = nn.ModuleList()
        for i in range(self.num_heads):
            heads.append(Attention(self.embed_dim, self.head_dim))
        
        return heads

class Attention(nn.Module):
    def __init__(self, embed_dim, out_dim ,eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.eps = eps
        self.fc_q = nn.Linear(embed_dim, out_dim)
        self.fc_k = nn.Linear(embed_dim, out_dim)
        self.fc_v= nn.Linear(embed_dim, out_dim)

    def forward(self, x):
        Q = self.fc_q(x) # [batch_size, num_patches, out_dim]
        K = self.fc_k(x) # [batch_size, num_patches, out_dim]
        V = self.fc_v(x) # [batch_size, num_patches, out_dim]

        attn = Q @ K.transpose(-2,-1) # [batch_size, num_patches, out_dim] *  [batch_size, out_dim , num_patches] = \
        # [batch_size, num_patches, num_patches]
        scaled_attn = attn / (torch.sqrt(torch.Tensor(self.out_dim)) + self.eps)
        scaled_attn_weights = F.softmax(scaled_attn, dim=-1)

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
    x = torch.randn(2, 2, 2)

    ln = LayerNormalization(2)
    x_norm = ln(x)

    print(f"Shape before norm: {x.shape}")
    print(f"shape after norm: {x_norm.shape}")

    # x = torch.randn(2, 2, 2)
    print()

    attn = Attention(2,2)
    x_attn = attn(x)

    print(f"Shape before attn: {x.shape}")
    print(f"shape after attn: {x_attn.shape}")
    print()

    x = torch.randn(2, 2, 2)
    start = time.time()
    attn = MultiHeadedAttention(2, 2)
    x_attn = attn(x)
    end = time.time()
    time_taken_01 = end-start
    print("Conceptual MultiHeadedAttention")
    print(f"Shape before attn: {x.shape}")
    print(f"Shape after attn: {x_attn.shape}")
    print(f"Time taken: {time_taken_01:5f}")
    print()
    start = time.time()
    attn = EfficientMultiHeadedAttention(2, 2)
    x_attn = attn(x)
    end = time.time()
    time_taken_02 = end-start
    print("Efficient MultiHeadedAttention")
    print(f"Shape before attn: {x.shape}")
    print(f"Shape after attn: {x_attn.shape}")
    print(f"Time taken: {time_taken_02:5f}")

    print(f"Speed up: {time_taken_01/time_taken_02}x faster") # Approx 3x faster