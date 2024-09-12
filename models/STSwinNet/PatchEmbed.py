import torch
import torch.nn as nn
import torch.nn.functional as F
from ..submodules import ResidualBlock

class residual_feature_generator(nn.Module):
    res_block = ResidualBlock
    def __init__(self, dim):
        super(residual_feature_generator, self).__init__()
        self.dim = dim
        self.resblock1 = self.res_block(dim, dim, 1, norm='BN')
        self.resblock2 = self.res_block(dim, dim, 1, norm='BN')
        self.resblock3 = self.res_block(dim, dim, 1, norm='BN')
        self.resblock4 = self.res_block(dim, dim, 1, norm='BN')

    def forward(self, x):
        out = self.resblock1(x)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        return out



class feature_generator(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(feature_generator, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels=dim,
                               out_channels=dim,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(in_channels=dim,
                               out_channels=dim,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=(kernel_size - 1) // 2)
        self.conv3 = nn.Conv2d(in_channels=dim,
                               out_channels=dim,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=(kernel_size - 1) // 2)
        self.conv4 = nn.Conv2d(in_channels=dim,
                               out_channels=dim,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(dim)
        self.bn3 = nn.BatchNorm2d(dim)
        self.bn4 = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn3(self.conv3(out)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn4(self.conv4(out)), negative_slope=0.01, inplace=False)
        return out




class PatchEmbedLocalGlobal(nn.Module):
    def __init__(self,   img_size= (240,320),patch_size=(2, 4, 4),  in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        patches_resolution = [img_size[0] // patch_size[1], img_size[1] // patch_size[2]]
        self.patches_resolution = patches_resolution
        self.in_chans = in_chans #T
        self.embed_dim = embed_dim

        self.num_blocks = self.in_chans // patch_size[0]

        self.head = nn.Conv2d(in_chans // self.num_blocks, embed_dim // 2, kernel_size=3, stride=1, padding=1)

        self.global_head = nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=1, padding=1)
        # self.chunk_head = nn.Conv2d(in_chans//2, embed_dim // 2, kernel_size=3, stride=1, padding=1)

        self.residual_encoding = residual_feature_generator(embed_dim // 2)
        self.global_residual_encoding = residual_feature_generator(embed_dim // 2)
        # self.chunk_residual_encoding = residual_feature_generator(embed_dim // 2)
        #spacial patch
        self.proj = nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=patch_size[1:], padding=1)
        self.global_proj = nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=patch_size[1:], padding=1)
        # self.chunk_proj = nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=patch_size[1:], padding=1)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        # patches_resolution = [224 // patch_size[1], 224 // patch_size[2]]
        # self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.num_blocks, patches_resolution[0], patches_resolution[1]))
        # trunc_normal_(self.absolute_pos_embed, std=.02)

    def forward(self, x):
        """Forward function."""
        # padding
        B, C, H, W = x.size()
        # if W % self.patch_size[2] != 0:
        #     x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        # if H % self.patch_size[1] != 0:
        #     x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        # if D % self.patch_size[0] != 0:
        #     x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        xs = x.chunk(self.num_blocks, 1)
        outs = []

        outi_global = self.global_head(x)
        outi_global = self.global_residual_encoding(outi_global)
        outi_global = self.global_proj(outi_global)


        for i in range(self.num_blocks):
            outi_local = self.head(xs[i])
            outi_local = self.residual_encoding(outi_local)
            outi_local = self.proj(outi_local)
            outi = torch.cat([outi_local, outi_global], dim=1)
            outi = outi.unsqueeze(2)
            outs.append(outi)

        out = torch.cat(outs, dim=2)  # B, 96, 4, H, W

        # x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = out.size(2), out.size(3), out.size(4)
            out = out.flatten(2).transpose(1, 2)
            out = self.norm(out)
            out = out.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return out, None

class PatchEmbedLocal(nn.Module):
    v2 = True #for numbin 10
    def __init__(self, img_size= (240,320),patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None, norm = None):
        super().__init__()
        self.patch_size = patch_size
        self.input_resolution = img_size
        self.patches_resolution = [img_size[0] // patch_size[1], img_size[1] // patch_size[2]]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.num_blocks = self.in_chans // patch_size[0]+1
        if self.v2:
            self.num_blocks = self.in_chans // patch_size[0]

        self.head = nn.Conv2d(patch_size[0], embed_dim, kernel_size=3, stride=1, padding=1)

        self.residual_encoding = residual_feature_generator(embed_dim)

        #spacial patch
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=patch_size[1:], padding=1)


        if norm_layer is not None:
            self.patch_norm = norm_layer(embed_dim)
        else:
            self.patch_norm = None

        # patches_resolution = [224 // patch_size[1], 224 // patch_size[2]]
        # self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.num_blocks, patches_resolution[0], patches_resolution[1]))
        # trunc_normal_(self.absolute_pos_embed, std=.02)

    def forward(self, x):
        """Forward function."""
        # padding
        # B, C, H, W = x.size()
        # if W % self.patch_size[2] != 0:
        #     x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        # if H % self.patch_size[1] != 0:
        #     x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        # if D % self.patch_size[0] != 0:
        #     x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        # xs = x.chunk(self.num_blocks, 1)
        outs = []

        for i in range(self.num_blocks):
            outi = self.head(x[i])
            outi = self.residual_encoding(outi)
            outi = self.proj(outi)
            outi = outi.unsqueeze(2)
            outs.append(outi)

        out = torch.cat(outs, dim=2)  # B, 96, 4, H, W

        # x = self.proj(x)  # B C D Wh Ww
        if self.patch_norm is not None:
            D, Wh, Ww = out.size(2), out.size(3), out.size(4)
            out = out.flatten(2).transpose(1, 2)
            out = self.patch_norm(out)
            out = out.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return out, None

    def extra_repr(self) -> str:
        return f"num_blocks={self.num_blocks}, patches_resolution={self.patches_resolution}"

    def flops(self):
        flops = 0
        #head
        flops += self.input_resolution[0] * self.input_resolution[1] * self.patch_size[0] * self.embed_dim * 3 * 3 * self.num_blocks
        #residual_encoding

        flops +=  self.input_resolution[0] * self.input_resolution[1] * self.embed_dim * self.embed_dim * 3 * 3 * 4 * 2 * self.num_blocks

        # project
        flops += self.patches_resolution[0] * self.patches_resolution[1] * self.embed_dim * self.embed_dim * 3 * 3 * self.num_blocks

        return flops

class PatchEmbedLocal_Conv(nn.Module):
    v2 = True #for numbin 10
    def __init__(self, img_size= (240,320),patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None, norm = None):
        super().__init__()
        self.patch_size = patch_size
        patches_resolution = [img_size[0] // patch_size[1], img_size[1] // patch_size[2]]
        self.patches_resolution = patches_resolution
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.num_blocks = self.in_chans // patch_size[0]+1
        if self.v2:
            self.num_blocks = self.in_chans // patch_size[0]

        self.head = nn.Conv2d(patch_size[0], embed_dim//2, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1)

        self.residual_encoding = residual_feature_generator(embed_dim)

        #spacial patch
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=patch_size[1:], padding=1)


        if norm_layer is not None:
            self.patch_norm = norm_layer(embed_dim)
        else:
            self.patch_norm = None

        # patches_resolution = [224 // patch_size[1], 224 // patch_size[2]]
        # self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.num_blocks, patches_resolution[0], patches_resolution[1]))
        # trunc_normal_(self.absolute_pos_embed, std=.02)

    def forward(self, x):
        """Forward function."""
        # padding
        # B, C, H, W = x.size()
        # if W % self.patch_size[2] != 0:
        #     x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        # if H % self.patch_size[1] != 0:
        #     x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        # if D % self.patch_size[0] != 0:
        #     x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        # xs = x.chunk(self.num_blocks, 1)
        outs = []
        for i in range(self.num_blocks):
            outi = self.head(x[i])
            outi = self.conv(outi)
            outi = self.residual_encoding(outi)
            outi = self.proj(outi)
            outi = outi.unsqueeze(2)
            outs.append(outi)

        out = torch.cat(outs, dim=2)  # B, 96, 4, H, W

        # x = self.proj(x)  # B C D Wh Ww
        if self.patch_norm is not None:
            D, Wh, Ww = out.size(2), out.size(3), out.size(4)
            out = out.flatten(2).transpose(1, 2)
            out = self.patch_norm(out)
            out = out.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return out, None

    def extra_repr(self) -> str:
        return f"num_blocks={self.num_blocks}, patches_resolution={self.patches_resolution}"

