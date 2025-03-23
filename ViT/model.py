"""
Vision Transformer (ViT) implementation based on the paper:
"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
https://arxiv.org/abs/2010.11929
"""

# imports
import torch
import torch.nn as nn
import torchvision
import math


# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self,
                img_size=224,
                patch_size = 16,
                in_channels = 3,
                embedding_dim = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches = (img_size//patch_size)**2

        # 1. images -> patches -> linear proj on each patch
        self.proj = nn.Conv2d(in_channels = in_channels,
                             out_channels = embedding_dim,
                             kernel_size = patch_size,
                             stride = patch_size,
                             padding=0)
        self.flatten = nn.Flatten(2,3)
        # 2. adding cls(class) token
        self.cls_token = nn.Parameter(torch.randn(1,1,embedding_dim),
                                     requires_grad= True) # by default-> true
        # 3. positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches+1,embedding_dim),
                                         requires_grad= True)

    def forward(self, x):
        
        x = self.proj(x)
        x = x.flatten(2,3)
        x = x.transpose(1,2)
        # print(f'Patched image shape: {x.shape}')

        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1,-1)
        
        # prepend cls token
        x = torch.cat((cls_token, x), dim=1)
        # print(f'patch embedding with cls token shape: {x.shape}')

        # add pos
        x = x + self.pos_embedding
        # print(f'patch + cls + pos shape: {x.shape}')
        return x


# Multi Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self,
                 embedding_dim = 768,
                 num_heads = 12,
                 dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim =  embedding_dim // num_heads

        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):

        # batch_size, num_patches+1, embedding_dim = x.shape
        batch_size, num_patches, embedding_dim = x.shape
        # print(f'input to multi head attention: {x.shape}')

        qkv = self.qkv_proj(x).reshape(batch_size, num_patches, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0,2,1,3) # [batch_size, num_heads, num_patches, dim]
        q,k,v = qkv.chunk(3, dim=-1)

        # scaled dot-product
        attn = (torch.matmul(q, k.transpose(-2,-1)))/ math.sqrt(self.head_dim)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v)
        x = x.permute(0,2,1,3) # [batch_size, num_patches, num_heads, dim]
        x = x.reshape(batch_size, num_patches, embedding_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# MLP
class MLP(nn.Module):
    def __init__(self,
                 embedding_dim=768,
                 mlp_size=3072,
                 dropout=0.):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp_size = mlp_size

        self.mlp = nn.Sequential(
            nn.Linear(in_features= embedding_dim,
                     out_features = mlp_size),
            nn.GELU(),
            nn.Dropout(p = dropout),
            nn.Linear(in_features = mlp_size,
                     out_features = embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
    

# Transformer Encoder
class TransformerBlock(nn.Module):
    def __init__(self,
               embedding_dim = 768,
               num_heads = 12,
               mlp_size=3072,
               mlp_dropout = 0.1,
               attn_dropout = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.mlp_size = mlp_size
        self.mlp_dropout = mlp_dropout
        

        self.layer_norm1 = nn.LayerNorm(normalized_shape = embedding_dim)
        self.attn = MultiHeadAttention(embedding_dim=embedding_dim,
                                       num_heads=num_heads,
                                       dropout = attn_dropout
                                      )
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim=embedding_dim,
                       mlp_size=mlp_size,
                       dropout=mlp_dropout)


    def forward(self,x):
        # layer norm and attn with residual conn
        x = x + self.attn(self.layer_norm1(x))

        # layer norm and mlp with residual conn
        x = x + self.mlp(self.layer_norm2(x))
        return x
    

# Vision Transformer
class ViT(nn.Module):
    def __init__(self,
                 img_size = 224,
                 patch_size = 16,
                 in_channels = 3,
                 num_classes = 10,
                 embedding_dim = 768,
                 mlp_size = 3072,
                 num_heads = 12,
                 attn_dropout = 0.1,
                 mlp_dropout = 0.1,
                 num_transformer_layer = 12):
        super().__init__()
    
        # patch embedding
        self.patch_embed = PatchEmbedding(
            img_size = img_size,
            patch_size = patch_size,
            in_channels = in_channels,
            embedding_dim = embedding_dim
        )
    
        # transformer encoder
        self.transformer_encoder = nn.Sequential(*[
            TransformerBlock(embedding_dim = embedding_dim,
                            num_heads = num_heads,
                            mlp_size = mlp_size,
                            mlp_dropout = mlp_dropout)
            for _ in range(num_transformer_layer)
        ])
    
        # classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
            x = self.patch_embed(x)
            x = self.transformer_encoder(x)
            x = self.classifier(x[:,0]) # only cls token
            return x
    


if __name__ == "__main__":

    batch_size = 4
    img_size = 224
    in_channels = 3
    num_classes = 10
    
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    
    model = ViT(img_size= img_size,
            patch_size = 16,
            in_channels = in_channels,
            num_classes = num_classes,
            )
    
    #forward pass
    output = model(x)
    
    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")