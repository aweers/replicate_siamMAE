# following ViT: https://arxiv.org/abs/2010.11929
# Attention is all you need (Transformer): https://arxiv.org/abs/1706.03762
# N: number of patches
# D: latent vector size

import torch
import torch.nn as nn
from pos_embed import get_2d_sincos_pos_embed

# LayerNorm: https://arxiv.org/abs/1607.06450
class LayerNorm(nn.Module):
    def __init__(self, D):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(D))
        self.bias = nn.Parameter(torch.zeros(D))
        self.eps = 1e-6

    def forward(self, x):
        # x.shape: (batch, N, D)
        mu = torch.mean(x, dim=-2, keepdim=True)
        sigma = torch.std(x, dim=-2, keepdim=True) + self.eps
        return ((x - mu) / sigma) * self.weight + self.bias

# ViT
# Transformer
class Attention(nn.Module):
    def __init__(self, D):
        super(Attention, self).__init__()
        self.sqrt_d = torch.sqrt(torch.tensor(D, dtype=torch.float32))
        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)

    def forward(self, x, z):
        # x.shape: (batch, N, D)
        # z.shape: (batch, N, D)
        q = self.Wq(x) # NxD
        k = self.Wk(z) # NxD
        v = self.Wv(z) # NxD

        s = torch.matmul(q, k.transpose(-2, -1)) # NxN
        attention = nn.functional.softmax(s / self.sqrt_D, dim=-1)
        return torch.matmul(attention, v)

# ViT
# Transformer
class MultiHeadAttention(nn.Module):
    def __init__(self, D, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.D = D
        self.sqrt_D = torch.sqrt(torch.tensor(D, dtype=torch.float32))
        self.K = D // heads

        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)
        self.Wo = nn.Linear(D, D)

    def forward(self, x, z):
        # x.shape: (batch, N, D)
        # z.shape: (batch, N, D)
        batch = x.shape[0]

        q = self.Wq(x).view(batch, -1, self.heads, self.K) # batch, N, heads, K
        k = self.Wk(z).view(batch, -1, self.heads, self.K) # batch, N, heads, K
        v = self.Wv(z).view(batch, -1, self.heads, self.K) # batch, N, heads, K

        q = q.transpose(1, 2) # batch, heads, N, K
        k = k.transpose(1, 2) # batch, heads, N, K
        v = v.transpose(1, 2) # batch, heads, N, K

        s = torch.matmul(q, k.transpose(-2, -1))
        attention = nn.functional.softmax(s / self.sqrt_D, dim=-1)

        y = torch.matmul(attention, v) # batch, heads, N, K
        y = y.transpose(1, 2).contiguous().view(batch, -1, self.D) # batch, N, D

        return self.Wo(y)

# ViT
class Encoder_Block(nn.Module):
    def __init__(self, D, heads, mlp_params):
        # mlp_params: [hidden_size, activation]
        super(Encoder_Block, self).__init__()
        self.D = D
        self.heads = heads
        self.attention = MultiHeadAttention(D, heads)
        self.norm1 = LayerNorm(D)
        self.norm2 = LayerNorm(D)

        self.mlp = nn.Sequential(
            nn.Linear(D, mlp_params[0]),
            mlp_params[1],
            nn.Linear(mlp_params[0], D)
        )

    def forward(self, x):
        # x.shape: (batch, N, D)
        x_norm = self.norm1(x)
        attention = self.attention(x_norm, x_norm)
        x = x + attention

        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x

class ViT_Encoder(nn.Module):
    def __init__(self, D, heads, mlp_params, num_blocks):
        super(ViT_Encoder, self).__init__()
        self.D = D
        self.heads = heads
        self.num_blocks = num_blocks
        self.encoder_blocks = nn.ModuleList([Encoder_Block(D, heads, mlp_params) for _ in range(num_blocks)])
        self.apply(self._init_weights)

    # Weight initialization is from https://github.com/facebookresearch/mae/blob/main/models_mae.py
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x.shape: (batch, N, D)
        for i in range(self.num_blocks):
            x = self.encoder_blocks[i](x)
        return x

# SiamMAE
class CrossSelfDecoderBlock(nn.Module):
    def __init__(self, D, heads, mlp_params):
        # mlp_params: [hidden_size, activation]
        super(CrossSelfDecoderBlock, self).__init__()
        self.D = D
        self.heads = heads
        self.cross_attention = MultiHeadAttention(D, heads)
        self.self_attention = MultiHeadAttention(D, heads)
        self.norm1 = LayerNorm(D)
        self.norm2 = LayerNorm(D)
        self.norm3 = LayerNorm(D)

        self.mlp = nn.Sequential(
            nn.Linear(D, mlp_params[0]),
            mlp_params[1],
            nn.Linear(mlp_params[0], D)
        )
    
    def forward(self, x, z):
        # x.shape: (batch, N, D)
        # z.shape: (batch, N, D)
        x_norm = self.norm1(x)
        cross_attention = self.cross_attention(x_norm, z)
        x = x + cross_attention

        x_norm = self.norm2(x)
        self_attention = self.self_attention(x_norm, x_norm)
        x = x + self_attention

        x_norm = self.norm3(x)
        x = x + self.mlp(x_norm)
        return x

class CrossSelfDecoder(nn.Module):
    def __init__(self, D, heads, mlp_params, num_blocks, pure_cross=True):
        super(CrossSelfDecoder, self).__init__()
        self.D = D
        self.heads = heads
        self.num_blocks = num_blocks
        self.pure_cross = pure_cross
        self.decoder_blocks = nn.ModuleList([CrossSelfDecoderBlock(D, heads, mlp_params) for _ in range(num_blocks)])

    def forward(self, x, z=None):
        if z is None:
            z = x
        for i in range(self.num_blocks):
            if i == 0 or self.pure_cross:
                x = self.decoder_blocks[i](x, z)
            else:
                x = self.decoder_blocks[i](x, x)
        return x

# SiamMAE for [CLS] token
# Transformer for positional encoding
class Embedding(nn.Module):
    # linear projection + positional encoding (fixed, sine and cosine)
    def __init__(self, D, patch_size, channel, num_patches, masking):
        super(Embedding, self).__init__()
        self.D = D
        self.patch_size = patch_size
        self.channel = channel
        self.masking = masking

        # embedding for all patches
        self.e = nn.Linear(patch_size * patch_size * channel, D)
        nn.init.xavier_uniform_(self.e.weight)
        # unembedding for all patches
        self.u = nn.Linear(D, patch_size * patch_size * channel)
        nn.init.xavier_uniform_(self.u.weight)


        self.decoder_embed = nn.Linear(D, D, bias=True)
        nn.init.xavier_uniform_(self.decoder_embed.weight)

        # positional embedding (fixed, sine and cosine)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, D), requires_grad=False)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, D), requires_grad=False)

        # Positional embedding is from https://github.com/facebookresearch/mae/blob/main/models_mae.py
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
    def embedding(self, x):
        # x.shape: (batch, channel, H, W)
        batch = x.shape[0]

        # split into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(batch, -1, self.patch_size * self.patch_size * self.channel) # batch, N, patch_size * patch_size * channel

        # embed patches with positional encoding
        patches = self.e(patches) + self.pos_embed[:, :patches.shape[1], :] # batch, N, D
        
        return patches
    
    def unembedding(self, x):
        # x.shape: (batch, num_patches, D)

        # unembed patches
        x = self.u(x)

        # reshape to: (batch, channel, H, W)
        batch, num_patches, _ = x.shape
        H = W = int((num_patches)**0.5) * self.patch_size
        x = x.view(batch, self.channel, H, W)

        return x
    
    def decoder_embedding(self, x, mask_type, shuffled_indices, skip):
        x = self.decoder_embed(x)
        x = self.masking.unmask(x, mask_type, shuffled_indices, skip)

        x += self.decoder_pos_embed

        return x

# MAE
class Masking(nn.Module):
    def __init__(self, D):
        super(Masking, self).__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, D))
    
    def mask(self, embeddings, mask_ratio, mask_type):
        # mask patches
        skip = int(embeddings.shape[1] * mask_ratio)
        if mask_type == 'random':
            # shuffle patches and save shuffled indices
            shuffled_indices = torch.randperm(embeddings.shape[1])
            embeddings = embeddings[:, shuffled_indices, :]
            embeddings = embeddings[:, :-skip, :]
        else:
            raise Exception('Invalid mask type')
        
        return embeddings, shuffled_indices, skip
    
    def mask_deterministic(self, embeddings, shuffled_indices, skip):
        # mask patches
        embeddings = embeddings[:, shuffled_indices, :]
        embeddings = embeddings[:, :-skip, :]
        
        return embeddings
        
    
    def unmask(self, embeddings, mask_type, shuffled_indices, skip):
        # append learnable masked patches
        if skip > 0:
            embeddings = torch.cat((embeddings, self.mask_token.repeat(embeddings.shape[0], skip, 1)), dim=1)

        # unshuffle patches
        if mask_type == 'random':
            unshuffled_indices = torch.zeros((shuffled_indices.shape[0]), dtype=torch.long)
            unshuffled_indices[shuffled_indices] = torch.arange(shuffled_indices.shape[0])
            embeddings = embeddings[:, unshuffled_indices, :]
        else:
            raise Exception('Invalid mask type')
        
        return embeddings
    
    # method to create a mask of size (batch, channel, H, W) with 1s in the masked region and 0s in the unmasked region
    def create_mask(self, patches_per_dim, mask_type, shuffled_indices, skip, channel=3, H=224, W=224, device=None):
        width = patches_per_dim
        mask = torch.zeros((patches_per_dim*patches_per_dim))
        if mask_type == 'random':
            mask[shuffled_indices[-skip:]] = 1
        else:
            raise Exception('Invalid mask type')
        
        mask = mask.view(width, width)
        
        # resize to (batch, channel, H, W)
        mask = mask.repeat(channel, 1, 1).unsqueeze(0)
        mask = nn.functional.interpolate(mask, size=(H, W), mode='nearest')
        mask = mask.to(device)

        return mask