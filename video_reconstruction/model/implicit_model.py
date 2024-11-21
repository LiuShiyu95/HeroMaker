import torch
from torch import nn
import math
import tinycudann as tcnn


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class TranslationField(nn.Module):
    def __init__(self, D=6, W=128,
                in_channels_w=8, in_channels_xyz=34,
                skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (2+2*8*2=34 by default)
        in_channels_w: number of channels for warping channels
        skips: add skip connection in the Dth layer
        """
        super(TranslationField, self).__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_w = in_channels_w
        self.typ = "translation"

        # encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz+self.in_channels_w, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz+self.in_channels_w, W)
            else:
                layer = nn.Linear(W, W)
            init_weights(layer)
            layer = nn.Sequential(layer, nn.ReLU(True))
            # init the models
            setattr(self, f"warping_field_xyz_encoding_{i+1}", layer)
        out_layer = nn.Linear(W, 2)
        nn.init.zeros_(out_layer.bias)
        nn.init.uniform_(out_layer.weight, -1e-4, 1e-4)
        self.output = nn.Sequential(out_layer)

    def forward(self, x):
        """
        Encodes input xyz to warp field for points

        Inputs:
            x: (B, self.in_channels_xyz)
               the embedded vector of position and direction
        Outputs:
            t: warping field
        """
        input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"warping_field_xyz_encoding_{i+1}")(xyz_)

        t = self.output(xyz_)

        return t


class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True, identity=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.annealed = False
        self.identity = identity
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        if self.identity:
            out = [x]
        else:
            out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class AnnealedEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs, annealed_step, annealed_begin_step=0, logscale=True, identity=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(AnnealedEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.annealed = True
        self.annealed_step = annealed_step
        self.annealed_begin_step = annealed_begin_step
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        self.index = torch.linspace(0, N_freqs-1, N_freqs)
        self.identity = identity

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x, step):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        if self.identity:
            out = [x]
        else:
            out = []

        if self.annealed_begin_step == 0:
            # calculate the w for each freq bands
            alpha = self.N_freqs * step / float(self.annealed_step)
        else:
            if step <= self.annealed_begin_step:
                alpha = 0
            else:
                alpha = self.N_freqs * (step - self.annealed_begin_step) / float(
                    self.annealed_step)

        for j, freq in enumerate(self.freq_bands):
            w = (1 - torch.cos(
                math.pi * torch.clamp(alpha - self.index[j], 0, 1))) / 2
            for func in self.funcs:
                out += [w * func(freq*x)]

        return torch.cat(out, -1)


class AnnealedHash(nn.Module):
    def __init__(self, in_channels, annealed_step, annealed_begin_step=0, identity=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(AnnealedHash, self).__init__()
        self.N_freqs = 16
        self.in_channels = in_channels
        self.annealed = True
        self.annealed_step = annealed_step
        self.annealed_begin_step = annealed_begin_step

        self.index = torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.identity = identity

        self.index_2 = self.index.view(-1, 1).repeat(1, 2).view(-1)

    def forward(self, x_embed, step):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """

        if self.annealed_begin_step == 0:
            # calculate the w for each freq bands
            alpha = self.N_freqs * step / float(self.annealed_step)
        else:
            if step <= self.annealed_begin_step:
                alpha = 0
            else:
                alpha = self.N_freqs * (step - self.annealed_begin_step) / float(
                    self.annealed_step)

        w = (1 - torch.cos(math.pi * torch.clamp(alpha * torch.ones_like(self.index_2) - self.index_2, 0, 1))) / 2

        out = x_embed * w.to(x_embed.device)

        return out


class ImplicitVideo(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=34,
                 skips=[4],
                 out_channels=3,
                 sigmoid_offset=0):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*8*2=51 by default)
        skips: add skip connection in the Dth layer

        ------ for nerfies ------
        encode_warp: whether to encode warping
        in_channels_w: dimension of warping embeddings
        """
        super(ImplicitVideo, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.skips = skips
        self.in_channels_xyz = self.in_channels_xyz
        self.sigmoid_offset = sigmoid_offset

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+self.in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            init_weights(layer)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)
        init_weights(self.xyz_encoding_final)

        # output layers
        self.rgb = nn.Sequential(nn.Linear(W, out_channels))

        self.rgb.apply(init_weights)

    def forward(self, x):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz)
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        input_xyz = x

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        out = self.rgb(xyz_encoding_final)

        out = torch.sigmoid(out) - self.sigmoid_offset

        return out


class ImplicitVideo_Hash(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = tcnn.Encoding(n_input_dims=2,
                                     encoding_config=config["encoding"])
        self.decoder = tcnn.Network(n_input_dims=self.encoder.n_output_dims +
                                    2,
                                    n_output_dims=3,
                                    network_config=config["network"])

    def forward(self, x):
        input = x
        input = self.encoder(input)
        input = torch.cat([x, input], dim=-1)
        weight = torch.ones(input.shape[-1], device=input.device).cuda()
        x = self.decoder(weight * input)
        return x


class Deform_Hash3d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = tcnn.Encoding(n_input_dims=3,
                                     encoding_config=config["encoding_deform3d"])
        self.decoder = tcnn.Network(n_input_dims=self.encoder.n_output_dims + 3,
                                    n_output_dims=2,
                                    network_config=config["network_deform"])

    def forward(self, x, step=0, aneal_func=None):
        input = x
        input = self.encoder(input)
        if aneal_func is not None:
            input = torch.cat([x, aneal_func(input,step)], dim=-1)
        else:
            input = torch.cat([x, input], dim=-1)

        weight = torch.ones(input.shape[-1], device=input.device).cuda()
        x = self.decoder(weight * input) / 5

        return x


class Deform_Hash3d_Warp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Deform_Hash3d = Deform_Hash3d(config)

    def forward(self, xyt_norm, step=0,aneal_func=None):
        x = self.Deform_Hash3d(xyt_norm,step=step, aneal_func=aneal_func)

        return x