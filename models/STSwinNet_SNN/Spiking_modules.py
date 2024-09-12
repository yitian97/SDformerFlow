import torch
import torch.nn as nn
import random
from spikingjelly.activation_based import (
    surrogate,
    neuron,
    functional,
    base,
    layer,
)
from models.STSwinNet_SNN.Spiking_submodules import *

import torch.nn.functional as F
from spikingjelly.activation_based.model import sew_resnet

if torch.cuda.is_available():
    dev_0 = "cpu"
    dev_1 = "cpu"
    dev_2 = "cpu"
else:
    dev_0 = "cpu"
    dev_1 = "cpu"
    dev_2 = "cpu"


class Spiking_neuron(nn.Module):
    def __init__(
        self,
        num_steps,
        spike_norm = None,
        neuron_type = "plif",
        v_th=1.,
        v_reset=0,
        surrogate_fun=surrogate.ATan(),
        tau=2.,
        detach_reset=True
    ):
        super().__init__()
        assert neuron_type in ["lif", "if", "plif", "SLTTlif", "glif", "psn"]
        if neuron_type == "lif":
            self.spiking_neuron = neuron.LIFNode(
                v_threshold=v_th,
                v_reset=v_reset,
                surrogate_function=eval(surrogate_fun),
                tau=tau,
                detach_reset=detach_reset
            )

        elif neuron_type == "SLTTlif":
            self.spiking_neuron = SLTTLIFNode(
                v_threshold=v_th,
                v_reset=v_reset,
                surrogate_function=eval(surrogate_fun),
                tau=tau,
                detach_reset=detach_reset
            )



        elif neuron_type == "if":
            self.spiking_neuron = neuron.IFNode(
                v_threshold=v_th,
                v_reset=v_reset,
                surrogate_function=eval(surrogate_fun),
                detach_reset=detach_reset
            )



        elif neuron_type == "plif":
            self.spiking_neuron = neuron.ParametricLIFNode(
                v_threshold=v_th,
                v_reset=v_reset,
                surrogate_function=eval(surrogate_fun),
                init_tau=tau,
                detach_reset=detach_reset
            )

        elif neuron_type == "glif":
            self.spiking_neuron = GatedLIFNode(
                T=num_steps,
                init_v_subreset = None,
                init_tau=0.25,
                init_v_threshold =0.5,
                init_conduct =0.5,
                surrogate_function= eval(surrogate_fun)
            )

        elif neuron_type == "psn":
            self.spiking_neuron = PSN(
                T=num_steps,
                surrogate_function=eval(surrogate_fun)
            )

        else:
            raise "neuron type not in the list!"

    def forward(self, x):
        return self.spiking_neuron(x)

class SpikingNormLayer(nn.Module):
    '''
    spike batch normalization layer, multistep processing
    '''
    def __init__(
        self,
        out_channels,
        num_steps,
        norm = 'BN',
        v_th=1.

    ):
        super().__init__()
        self.num_steps = num_steps
        self.norm = norm
        num_groups = out_channels//16
        if self.norm == "BN":
            self.norm_layer = layer.BatchNorm2d(out_channels)
        if self.norm == "BN_notrack":
            self.norm_layer = layer.BatchNorm2d(out_channels, track_running_stats=False)
        if self.norm == "GN":
            self.norm_layer = layer.GroupNorm(num_groups, out_channels)
        if self.norm == "IN":
            self.norm_layer = layer.GroupNorm(out_channels, out_channels)
        if self.norm == "LN":
            self.norm_layer = layer.GroupNorm(1, out_channels)
        elif self.norm == "BNTT":
            self.norm_layer = nn.ModuleList(
                [nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.1, affine=True) for i in range(self.num_steps)])
        elif self.norm == "TDBN":
            self.norm_layer = layer.ThresholdDependentBatchNorm2d(alpha=1,v_th=v_th,num_features= out_channels)

    def forward(self, x):
        if self.norm is None:
            pass

        if self.norm == 'BNTT':
            out_norm = []
            for i in range(self.num_steps):
                x_step = self.norm_layer[i](x[i])
                x_step = x_step.unsqueeze(0)
                out_norm.append(x_step)
            x= torch.cat(out_norm,dim=0)
        else:
            x = self.norm_layer(x)
        return x

class MS_SepConv(nn.Module):
    """
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            dim,
            kernel_size=7,
            padding=3,
            expansion_ratio=2,
            act2_layer=nn.Identity,
            **spiking_kwargs
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.norm = spiking_kwargs["spike_norm"]
        bias = True if self.norm is None else False

        self.sn1 = Spiking_neuron(**spiking_kwargs)
        self.pwconv1 = nn.Sequential(layer.Conv2d(dim, med_channels, kernel_size=1, stride=1, bias=bias))
        if self.norm is not None:
            self.norm1 = SpikingNormLayer(med_channels, spiking_kwargs["num_steps"], self.norm,
                                          v_th=spiking_kwargs["v_th"])

        self.sn2 = Spiking_neuron(**spiking_kwargs)
        self.dwconv = nn.Sequential(layer.Conv2d(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        ))  # depthwise conv
        self.pwconv2 = nn.Sequential(layer.Conv2d(med_channels, dim, kernel_size=1, stride=1, bias=bias))
        if self.norm is not None:
            self.norm2 = SpikingNormLayer(dim, spiking_kwargs["num_steps"], self.norm,
                                          v_th=spiking_kwargs["v_th"])

    def forward(self, x):
        x = self.sn1(x)
        x = self.pwconv1(x)
        if self.norm is not None:
            x = self.norm1(x)
        x = self.sn2(x)
        x = self.dwconv(x)
        x = self.pwconv2(x)
        if self.norm is not None:
            x = self.norm2(x)
        return x

class MS_SpikingSepConvEncoderBlock(nn.Module):
    """
    SepConvolution layer with spiking neuron with MS shortcut. De
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        **spiking_kwargs
    ):
        super().__init__()

        self.norm = spiking_kwargs["spike_norm"]
        bias = True if self.norm is None else False

        self.SepConv = MS_SepConv(
            dim=in_channels,
            kernel_size=7,
            padding=3,
            expansion_ratio=2,
            **spiking_kwargs
            )
        self.sn1 = Spiking_neuron(**spiking_kwargs)
        self.conv1 = nn.Sequential(layer.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, groups=1, bias=bias
        ))
        if self.norm is not None:
            self.norm1 = SpikingNormLayer(out_channels, spiking_kwargs["num_steps"], self.norm, v_th=spiking_kwargs["v_th"])

        self.sn2 = Spiking_neuron(**spiking_kwargs)
        self.conv2 = nn.Sequential(layer.Conv2d(
            out_channels, in_channels, kernel_size=3, padding=1, groups=1, bias=bias
        ))
        # self.conv2 = RepConv(dim*mlp_ratio, dim)
        if self.norm is not None:
            self.norm2 = SpikingNormLayer(in_channels, spiking_kwargs["num_steps"], self.norm,
                                          v_th=spiking_kwargs["v_th"])

    def forward(self, x):
        x = self.SepConv(x) + x
        x_feat = x
        x = self.conv1(self.sn1(x))
        if self.norm is not None:
            x = self.norm1(x)
        x = self.conv2(self.sn2(x))
        if self.norm is not None:
            x = self.norm2(x)
        x = x_feat + x

        return x

class SpikingConvEncoderLayer(nn.Module):
    """
    Convolution layer with spiking neuron with SEW shortcut.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding = 1,
        spike_norm=None,
        **spiking_kwargs
    ):
        super().__init__()

        self.norm = spike_norm
        bias = True if self.norm is None else False
        self.conv = nn.Sequential(
            layer.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding= padding,
                bias=bias
                ),
            # self.norm_layer,
            # spiking_neuron,

        )


        if self.norm is not None:
            self.norm_layer = SpikingNormLayer(out_channels, spiking_kwargs["num_steps"], self.norm, v_th=spiking_kwargs["v_th"])

        self.sn = Spiking_neuron(**spiking_kwargs)



    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm_layer(x)
        out = self.sn(x)
        return out

class MS_SpikingConvEncoderLayer(nn.Module):
    """
    Convolution layer with spiking neuron with MS shortcut.
    no spike layer for the first layer
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding = 1,
        first_layer=True,
        spike_norm=None,
        **spiking_kwargs,

    ):
        super().__init__()
        self.first_layer = first_layer
        self.norm = spike_norm
        bias = True if self.norm is None else False
        if not self.first_layer:
            self.sn = Spiking_neuron(**spiking_kwargs)
        self.conv = nn.Sequential(
            layer.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding= padding,
                bias=bias
                ),
            # self.norm_layer,
            # spiking_neuron,

        )


        if self.norm is not None:
            self.norm_layer = SpikingNormLayer(out_channels, spiking_kwargs["num_steps"], self.norm, v_th=spiking_kwargs["v_th"])

    def forward(self, x):
        if not self.first_layer:
            x = self.sn(x)

        x = self.conv(x)
        if self.norm is not None:
            x = self.norm_layer(x)

        return x

class SpikingDecoderLayer(nn.Module):
    """
    Upsampling spiking layer to increase spatial resolution (x2) in a decoder.
    SEW shortcut
    """
    upsample_mode = "bilinear" # bilinear or nearest

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        spike_norm=None,
        scale = 2,
        **spiking_kwargs
    ):
        super().__init__()

        self.scale = scale
        self.norm = spike_norm
        bias = True if self.norm is None else False
        self.deconv = nn.Sequential(

            layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding = kernel_size // 2,bias = bias),

        )
        if self.norm is not None:
            self.norm_layer = SpikingNormLayer(out_channels, spiking_kwargs["num_steps"], self.norm, v_th=spiking_kwargs["v_th"])

        self.sn = Spiking_neuron(**spiking_kwargs)

    def forward(self, x):
        x_out=[]
        steps =x.shape[0]
        for i in range(steps):
            x_up = F.interpolate(x[i], scale_factor=self.scale, mode= self.upsample_mode, align_corners=False)
            x_up = x_up.unsqueeze(0)
            x_out.append(x_up)

        x_out = torch.cat(x_out,dim=0)
        # x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.deconv(x_out)
        if self.norm is not None:
            x = self.norm_layer(x)
        out = self.sn(x)

        return out

class SpikingTransposeDecoderLayer(nn.Module):
    """
    Upsampling spiking layer using transposed convolution to increase spatial resolution (x2) in a decoder.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        spike_norm=None,
        scale = 2,
        **spiking_kwargs
    ):
        super().__init__()

        self.scale = scale
        self.norm = spike_norm
        padding = kernel_size // 2
        bias = True if self.norm is None else False
        if scale == 2:
            self.deconv = nn.Sequential(

                layer.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=1,
                    bias=bias,
                )
            )
        elif scale == 4:
            self.deconv = nn.Sequential(

                layer.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    7,
                    stride=4,
                    padding=2,
                    output_padding=1,
                    bias=bias,
                )
            )
        if self.norm is not None:
            self.norm_layer = SpikingNormLayer(out_channels, spiking_kwargs["num_steps"], self.norm, v_th=spiking_kwargs["v_th"])

        self.sn = Spiking_neuron(**spiking_kwargs)



    def forward(self, x):

        x = self.deconv(x)
        if self.norm is not None:
            x = self.norm_layer(x)
        out = self.sn(x)

        return out

class MS_SpikingTransposeDecoderLayer(SpikingTransposeDecoderLayer):
    """
    Upsampling spiking layer using transposed convolution to increase spatial resolution (x2) in a decoder.
    MS shortcut
    """

    def forward(self, x):

        x = self.sn(x)
        x = self.deconv(x)
        if self.norm is not None:
            x = self.norm_layer(x)

        return x

class MS_SpikingSepTransposeDecoderLayer(nn.Module):
    """
    Upsampling spiking layer using separable transposed convolution to increase spatial resolution (x2) in a decoder.
    MS shortcut
    """


    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        spike_norm=None,
        scale = 2,
        **spiking_kwargs
    ):
        super().__init__()

        self.scale = scale
        self.norm = spike_norm
        padding = kernel_size // 2
        bias = True if self.norm is None else False
        self.sn1 = Spiking_neuron(**spiking_kwargs)
        self.deconv1 = nn.Sequential(

            layer.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
                bias=bias,
            )

        )
        if self.norm is not None:
            self.norm_layer1 = SpikingNormLayer(out_channels, spiking_kwargs["num_steps"], self.norm, v_th=spiking_kwargs["v_th"])


        self.sn2 = Spiking_neuron(**spiking_kwargs)
        self.deconv2 = nn.Sequential(

            layer.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=2,
                bias=bias,
            )

        )
        if self.norm is not None:
            self.norm_layer2 = SpikingNormLayer(out_channels, spiking_kwargs["num_steps"], self.norm, v_th=spiking_kwargs["v_th"])


    def forward(self, x):

        x = self.sn1(x)
        x = self.deconv1(x)
        if self.norm is not None:
            x = self.norm_layer1(x)
        x = self.sn2(x)
        x = self.deconv2(x)
        if self.norm is not None:
            x = self.norm_layer2(x)


        return x

class MS_SpikingDecoderLayer(SpikingDecoderLayer):
    """
    Upsampling spiking layer to increase spatial resolution (x2) in a decoder.
    MS shortcut
    """
    def forward(self, x):
        x_out = []
        steps = x.shape[0]
        for i in range(steps):
            x_up = F.interpolate(x[i], scale_factor=2, mode="bilinear", align_corners=False)
            x_up = x_up.unsqueeze(0)
            x_out.append(x_up)

        x_out = torch.cat(x_out, dim=0)
        # x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if self.sn is not None:
            x = self.sn(x_out)
        x = self.deconv(x)
        if self.norm is not None:
            x = self.norm_layer(x)
        return x

class SpikingPredLayer(nn.Module):
    """
    Convolution layer with spiking neuron.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        **spiking_kwargs
    ):
        super().__init__()
        padding = kernel_size // 2
        self.norm = None
        bias = True if self.norm is None else False
        self.conv = nn.Sequential(
            layer.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding= padding,
                bias=bias
                ),
            # self.norm_layer,
            # spiking_neuron,

        )




    def forward(self, x):
        x = self.conv(x)

        return x

class MS_SpikingPredLayer(nn.Module):
    """
    Convolution layer with spiking neuron.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        **spiking_kwargs
    ):
        super().__init__()
        padding = kernel_size // 2
        self.norm = None
        bias = True if self.norm is None else False
        self.sn = Spiking_neuron(**spiking_kwargs)
        self.conv = nn.Sequential(
            layer.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding= padding,
                bias=bias
                ),
            # self.norm_layer,
            # spiking_neuron,

        )





    def forward(self, x):
        x = self.sn(x)
        x = self.conv(x)

        return x

class MS_SpikingSepPredLayer(nn.Module):
    """
    SepConvolution layer with spiking neuron.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        **spiking_kwargs
    ):
        super().__init__()
        padding = kernel_size // 2
        self.norm = None
        bias = True if self.norm is None else False
        self.sn = Spiking_neuron(**spiking_kwargs)
        self.pwconv = nn.Sequential(
            layer.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=bias
                ),
            # self.norm_layer,
            # spiking_neuron,

        )
        self.dwconv = nn.Sequential(
            layer.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding= padding,
                groups=out_channels,
                bias=bias
                ),
            # self.norm_layer,
            # spiking_neuron,

        )




    def forward(self, x):
        x = self.sn(x)
        x = self.pwconv(x)
        x = self.dwconv(x)

        return x

class SpikingEmbeddingLayer(nn.Module):
    """
    Layer comprised of a Spiking convolution layer for patch embedding
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding = 1,
        norm=None,
        patch_resolution = (120,160),
        use_MS= False,
        **spiking_kwargs
    ):
        super().__init__()

        self.use_MS = use_MS
        self.norm = norm
        self.patch = patch_resolution
        bias = True if norm is None else False
        self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                bias=bias
                )

        if self.norm is not None:
            self.norm_layer = nn.BatchNorm2d(out_channels)


        self.sn = Spiking_neuron(**spiking_kwargs)

    def _forward_MS(self, x):
        T, B, C, H, W = x.shape

        x = self.sn(x)
        x = self.conv(x.flatten(0, 1))
        if self.norm is not None:
            x = self.norm_layer(x)
        x = x.reshape(T, B, -1, self.patch[0], self.patch[1]).contiguous()
        return x

    def _forward(self,x):
        T, B, C, H, W = x.shape
        out = self.conv(x.flatten(0, 1))
        #patch norm
        if self.norm is not None:
            out = self.norm_layer(out)
        out = out.reshape(T, B, -1, self.patch[0], self.patch[1]).contiguous()
        out = self.sn(out)


        return out

    def forward(self, x):
        if self.use_MS:
            out = self._forward_MS(x)
        else:
            out = self._forward(x)

        return out

class SpikingPEDLayer(nn.Module):
    """
    patch embedding with deformed shortcut
    spatial resolution is reduced by 2
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding = 1,
        norm=None,
        patch_resolution = (120,160),
        **spiking_kwargs
    ):
        super().__init__()

        self.norm = norm
        self.patch = patch_resolution
        bias = True if norm is None else False
        self.conv_res = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                padding=0,
                bias=bias
                )
        self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                bias=bias
                )

        if self.norm is not None:
            self.norm_layer = nn.BatchNorm2d(out_channels)


        self.sn = Spiking_neuron(**spiking_kwargs)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_res = self.conv_res(x.flatten(0, 1))
        x = self.sn(x)
        x = self.conv(x.flatten(0, 1))
        if self.norm is not None:
            x = self.norm_layer(x)
        x = (x+x_res).reshape(T, B, -1, self.patch[0], self.patch[1]).contiguous()
        return x

class SEWResBlock(nn.Module):
    """
    Spike-Element-Wise (SEW) residual block as it is described in the paper "Deep Residual Learning in Spiking Neural Networks".
    See https://arxiv.org/abs/2102.04159
    """

    def __init__(self,
                 in_channels: int,
                 out_channels,
                 stride=1,
                 connect_function='ADD',
                 spike_norm=None,
                 **spiking_kwargs
    ):
        super(SEWResBlock,self).__init__()

        self.norm = spike_norm
        bias = True if self.norm is None else False
        self.conv1 = nn.Sequential(layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=(3 - 1) // 2, bias=bias))
        self.conv2 = nn.Sequential(layer.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=bias))
        if self.norm is not None:
            self.norm1 = SpikingNormLayer(out_channels,self.norm,v_th=spiking_kwargs["v_th"])
            self.norm2 = SpikingNormLayer(out_channels,self.norm,v_th=spiking_kwargs["v_th"])
        self.sn1 = Spiking_neuron(**spiking_kwargs)
        self.sn2 = Spiking_neuron(**spiking_kwargs)
        self.connect_function = connect_function

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        if self.norm is not None:
            x = self.norm1(x)
        out1 = self.sn1(x)
        x = self.conv2(out1)
        if self.norm is not None:
            x = self.norm2(x)
        out = self.sn2(x)

        if self.connect_function == 'ADD':
            out = out + identity
        elif self.connect_function == 'MUL' or self.connect_function == 'AND':
            out = out * identity
        elif self.connect_function == 'OR':
            out = surrogate.ATan(spiking=True)(out + identity)
        elif self.connect_function == 'NMUL':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out

class MS_ResBlock(nn.Module):
    """
    SpikingResblock with Membrane potential shortcut
    "Spike-driven Transformer" NIPS 2023
    """

    def __init__(self,
                 in_channels: int,
                 out_channels,
                 stride=1,
                 connect_function='ADD',
                 spike_norm=None,
                 **spiking_kwargs
    ):
        super(MS_ResBlock, self).__init__()

        self.norm = spike_norm
        bias = True if self.norm is None else False
        self.conv1 = nn.Sequential(layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=(3 - 1) // 2, bias=bias))
        self.conv2 = nn.Sequential(layer.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=bias))
        if self.norm is not None:
            self.norm1 = SpikingNormLayer(out_channels,self.norm,v_th=spiking_kwargs["v_th"])
            self.norm2 = SpikingNormLayer(out_channels,self.norm,v_th=spiking_kwargs["v_th"])
        self.sn1 = Spiking_neuron(**spiking_kwargs)
        self.sn2 = Spiking_neuron(**spiking_kwargs)
        self.connect_function = connect_function

    def forward(self, x):

        identity = x

        x = self.sn1(x)
        x = self.conv1(x)
        if self.norm is not None:
            x = self.norm1(x)

        x = self.sn2(x)
        x = self.conv2(x)
        if self.norm is not None:
            out = self.norm2(x)


        if self.connect_function == 'ADD':
            out = out + identity
        elif self.connect_function == 'MUL' or self.connect_function == 'AND':
            out = out * identity
        elif self.connect_function == 'OR':
            out = surrogate.ATan(spiking=True)(out + identity)
        elif self.connect_function == 'NMUL':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out

class spiking_residual_feature_generator(nn.Module):
    """
    spiking version of the residual feature generator
    SEW residual block
    """

    res_block_type = SEWResBlock

    def __init__(self, dim, norm, num_resblocks=4, cnt_fun='ADD',**spiking_kwargs):
        super(spiking_residual_feature_generator, self).__init__()
        self.dim = dim
        self.resblocks = nn.ModuleList()
        self.num_resblocks = num_resblocks
        for i in range(self.num_resblocks):
            self.resblocks.append(
                self.res_block_type(
                    dim,
                    dim,
                    stride=1,
                    spike_norm=norm,
                    connect_function=cnt_fun,
                    **spiking_kwargs

                )

            )


    def forward(self, x):
        for i in range(self.num_resblocks):
            x = self.resblocks[i](x)
        return x

class MS_spiking_residual_feature_generator(spiking_residual_feature_generator):
    """
    spiking version of the residual feature generator
    MS residual block
    """
    res_block_type = MS_ResBlock

class Spiking_PatchEmbedLocal(nn.Module):
    use_MS = False

    def __init__(self, img_size= (240,320),patch_size=(2, 4, 4), in_chans=2, embed_dim=96, patch_norm=None, norm = None, spiking_proj = False, spike_norm = None,  **spiking_kwargs):
        super().__init__()

        self.patch_size = patch_size
        patches_resolution = [img_size[0] // patch_size[2], img_size[1] // patch_size[3]]
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # self.num_blocks = in_chans // patch_size[0]
        self.num_steps = spiking_kwargs['num_steps']
        # assert self.num_blocks ==self.num_steps

        self.spike_norm = spike_norm #norm for resnet
        self.head = SpikingConvEncoderLayer(
            in_chans // (self.num_steps - 1),
            self.embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            spike_norm=self.spike_norm,
            **spiking_kwargs
        )
        if self.use_MS:
            self.residual_encoding = MS_spiking_residual_feature_generator(
                dim=self.embed_dim,
                norm=self.spike_norm,
                num_resblocks=4,
                cnt_fun='ADD',
                **spiking_kwargs
            )


        else:

            self.residual_encoding = spiking_residual_feature_generator(
                dim=self.embed_dim,
                norm=self.spike_norm,
                num_resblocks=4,
                cnt_fun='ADD',
                **spiking_kwargs
            )

        self.spike_proj_flag = spiking_proj
        if self.spike_proj_flag:
            self.proj = SpikingEmbeddingLayer(embed_dim, embed_dim, kernel_size=3, stride=patch_size[2:], padding=1, norm = self.spike_norm, patch_resolution = self.patches_resolution, use_MS=self.use_MS, **spiking_kwargs)

        else:
            self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=patch_size[2:], padding=1)



    def forward(self, x):
        """Forward function."""

        xs = self.head(x)
        # spiking_rates["head"] = torch.mean((xs != 0).float())

        # out, spiking_rates["res"] = self.residual_encoding(xs)
        out = self.residual_encoding(xs)


        # out_plot = out.detach()
        # plot_2d_feature_map(out_plot[0,0,:,...])
        if self.spike_proj_flag:
            out = self.proj(out)  # T, B, C, H, W

        else:
            outs = []
            for i in range(self.num_blocks):
                outi = self.proj(out[i])
                outi = outi.unsqueeze(2)
                outs.append(outi)

            out = torch.cat(outs, dim=2)  # B, 96, B*P, H, W

            if self.patch_norm is not None:
                D, Wh, Ww = out.size(2), out.size(3), out.size(4)
                out = out.flatten(2).transpose(1, 2)
                out = self.patch_norm(out)
                out = out.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)


        return out

    def extra_repr(self) -> str:
        return f" num_steps={self.num_steps}, patches_resolution={self.patches_resolution}"

class Spiking_PatchEmbed_sfn(nn.Module):
    '''
    spiking patch embedding layer with 4 channel input as spike flow net
    SEW shortcut
    '''
    use_MS = False
    num_res = 2

    def __init__(self, img_size= (240,320),patch_size=(2, 4, 4), in_chans=2, embed_dim=96, patch_norm=None, norm = None, spiking_proj = False, spike_norm = None,  **spiking_kwargs):
        super().__init__()

        self.patch_size = patch_size
        patches_resolution = [img_size[0] // patch_size[2], img_size[1] // patch_size[3]]
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # self.num_blocks = in_chans // patch_size[0]
        self.num_bins = in_chans
        self.num_steps = spiking_kwargs['num_steps']
        self.num_ch = in_chans *2// self.num_steps

        # assert self.num_blocks ==self.num_steps

        self.spike_norm = spike_norm #norm for resnet

        if self.use_MS:
            self.head = MS_SpikingConvEncoderLayer(
                self.num_ch,
                self.embed_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                first_layer=True,
                spike_norm=self.spike_norm,
                **spiking_kwargs
            )
            self.residual_encoding = MS_spiking_residual_feature_generator(
                dim=self.embed_dim,
                norm=self.spike_norm,
                num_resblocks=self.num_res,
                cnt_fun='ADD',
                **spiking_kwargs
            )


        else:
            self.head = SpikingConvEncoderLayer(
                self.num_ch,
                self.embed_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                spike_norm=self.spike_norm,
                **spiking_kwargs
            )
            self.residual_encoding = spiking_residual_feature_generator(
                dim=self.embed_dim,
                norm=self.spike_norm,
                num_resblocks=self.num_res,
                cnt_fun='ADD',
                **spiking_kwargs
            )

        self.spike_proj_flag = spiking_proj
        # self.proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        # self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size[2:], stride=patch_size[2:])
        if self.spike_proj_flag:
            self.proj = SpikingEmbeddingLayer(embed_dim, embed_dim, kernel_size=3, stride=patch_size[2:], padding=1, norm = self.spike_norm, patch_resolution = self.patches_resolution, use_MS=self.use_MS, **spiking_kwargs)

        else:
            self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=patch_size[2:], padding=1)



    def forward(self, x):
        """Forward function."""
        # padding
        if x.size(1) > self.num_bins:
            x = x[:, :self.num_bins, :, :, :]

        event_reprs = x.permute(0, 2, 3, 4, 1)

        new_event_reprs = torch.zeros(event_reprs.size(0), self.num_ch, event_reprs.size(2), event_reprs.size(3),
                                      self.num_steps).to(event_reprs.device)

        for i in range(self.num_ch):
            start, end = i//2 * self.num_steps, (i//2 + 1) * self.num_steps
            new_event_reprs[:, i, :, :, :] = event_reprs[:, i % 2, :, :, start:end]

        x = new_event_reprs.permute(4, 0, 1, 2, 3)


        # spiking_rates = {"input": torch.mean((x != 0).float())}

        xs = self.head(x)
        # spiking_rates["head"] = torch.mean((xs != 0).float())

        # out, spiking_rates["res"] = self.residual_encoding(xs)
        out = self.residual_encoding(xs)


        # out_plot = out.detach()
        # plot_2d_feature_map(out_plot[0,0,:,...])
        if self.spike_proj_flag:
            out = self.proj(out)  # T, B, C, H, W

        else:
            outs = []
            for i in range(self.num_blocks):
                outi = self.proj(out[i])
                outi = outi.unsqueeze(2)
                outs.append(outi)

            out = torch.cat(outs, dim=2)  # B, 96, B*P, H, W

            if self.patch_norm is not None:
                D, Wh, Ww = out.size(2), out.size(3), out.size(4)
                out = out.flatten(2).transpose(1, 2)
                out = self.patch_norm(out)
                out = out.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)


        return out

    def extra_repr(self) -> str:
        return f" num_steps={self.num_steps}, patches_resolution={self.patches_resolution}"

class MS_Spiking_PatchEmbed_sfn(Spiking_PatchEmbed_sfn):
    '''
    spiking patch embedding layer with 4 channel input as spike flow net
    MS shortcut
    '''
    use_MS = True
    num_res = 2

class Spiking_PatchEmbed_Conv(nn.Module):
    '''
    spiking patch embedding layer with 4 channel input as spike flow net
    extra convolutional layer for downsampling
    SEW shortcut
    '''
    use_MS = False
    num_res = 2

    def __init__(self, img_size= (240,320),patch_size=(2, 4, 4), in_chans=2, embed_dim=96, patch_norm=None, norm = None, spiking_proj = False, spike_norm = None,  **spiking_kwargs):
        super().__init__()

        self.patch_size = patch_size
        patches_resolution = [img_size[0] // patch_size[2] // 2, img_size[1] // patch_size[3] // 2]
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # self.num_blocks = in_chans // patch_size[0]
        self.num_bins = in_chans
        self.num_steps = spiking_kwargs['num_steps']
        self.num_ch = in_chans *2// self.num_steps

        # assert self.num_blocks ==self.num_steps

        self.spike_norm = spike_norm #norm for resnet
        self.head = SpikingConvEncoderLayer(
            self.num_ch,
            self.embed_dim // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            spike_norm=self.spike_norm,
            **spiking_kwargs
        )
        if self.use_MS:

            self.conv = MS_SpikingConvEncoderLayer(
                self.embed_dim//2,
                self.embed_dim,
                kernel_size=3,
                stride=2,
                padding=3 // 2,
                spike_norm=self.spike_norm,
                **spiking_kwargs
            )
            self.residual_encoding = MS_spiking_residual_feature_generator(
                dim=self.embed_dim,
                norm=self.spike_norm,
                num_resblocks=4,
                cnt_fun='ADD',
                **spiking_kwargs
            )


        else:

            self.conv = SpikingConvEncoderLayer(
                self.embed_dim//2,
                self.embed_dim,
                kernel_size=3,
                stride=2,
                padding=3 // 2,
                spike_norm=self.spike_norm,
                **spiking_kwargs
            )
            self.residual_encoding = spiking_residual_feature_generator(
                dim=self.embed_dim,
                norm=self.spike_norm,
                num_resblocks=self.num_res,
                cnt_fun='ADD',
                **spiking_kwargs
            )


        self.spike_proj_flag = spiking_proj
        # self.proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        # self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size[2:], stride=patch_size[2:])
        if self.spike_proj_flag:
            self.proj = SpikingEmbeddingLayer(embed_dim, embed_dim, kernel_size=3, stride=patch_size[2:], padding=1, norm = self.spike_norm, patch_resolution = self.patches_resolution, use_MS=self.use_MS, **spiking_kwargs)

        else:
            self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=patch_size[2:], padding=1)



    def forward(self, x):
        """using nearest bins as channels"""

        if x.size(1) > self.num_bins:
            x = x[:, :self.num_bins, :, :, :]

        x = x.view([x.shape[0], -1] + list(x.shape[3:]))
        xs = x.chunk(self.num_steps, 1)
        x = torch.stack(list(xs), dim=1).permute(1, 0, 2, 3, 4)  #


        # spiking_rates = {"input": torch.mean((x != 0).float())}

        xs = self.head(x)
        # spiking_rates["head"] = torch.mean((xs != 0).float())

        # out, spiking_rates["res"] = self.residual_encoding(xs)
        xs = self.conv(xs)
        out = self.residual_encoding(xs)


        # out_plot = out.detach()
        # plot_2d_feature_map(out_plot[0,0,:,...])
        if self.spike_proj_flag:
            out = self.proj(out)  # T, B, C, H, W

        else:
            outs = []
            for i in range(self.num_blocks):
                outi = self.proj(out[i])
                outi = outi.unsqueeze(2)
                outs.append(outi)

            out = torch.cat(outs, dim=2)  # B, 96, B*P, H, W

            if self.patch_norm is not None:
                D, Wh, Ww = out.size(2), out.size(3), out.size(4)
                out = out.flatten(2).transpose(1, 2)
                out = self.patch_norm(out)
                out = out.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)


        return out

    def extra_repr(self) -> str:
        return f" num_steps={self.num_steps}, patches_resolution={self.patches_resolution}"

class MS_Spiking_PatchEmbed_Conv_Local(nn.Module):
    '''
    spiking patch embedding layer with 4 channel input as spike flow net
    extra convolutional layer for downsampling
    MS shortcut
    '''
    use_MS = True
    num_res = 2
    first_conv_k = 3

    def __init__(self, img_size= (240,320),patch_size=(2, 4, 4), in_chans=10, embed_dim=96, patch_norm=None, norm = None, spiking_proj = False, spike_norm = None,  **spiking_kwargs):
        super().__init__()

        self.patch_size = patch_size
        self.image_size = img_size
        self.patches_resolution = [img_size[0] // patch_size[2] // 2, img_size[1] // patch_size[3] // 2]
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # self.num_blocks = in_chans // patch_size[0]
        self.num_bins = in_chans
        self.num_steps = spiking_kwargs['num_steps']
        self.num_ch = 2
        self.num_blocks = self.num_bins // self.num_steps
        # assert self.num_blocks ==self.num_steps

        self.spike_norm = spike_norm #norm for resnet

        self.head = SpikingConvEncoderLayer(
            self.num_ch,
            self.embed_dim // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            spike_norm=self.spike_norm,
            **spiking_kwargs
        )

        if self.use_MS:
            self.conv = MS_SpikingConvEncoderLayer(
                self.embed_dim//2,
                self.embed_dim,
                kernel_size=self.first_conv_k,
                stride=2,
                padding=self.first_conv_k // 2,
                spike_norm=self.spike_norm,
                **spiking_kwargs
            )
            self.residual_encoding = MS_spiking_residual_feature_generator(
                dim=self.embed_dim,
                norm=self.spike_norm,
                num_resblocks=self.num_res,
                cnt_fun='ADD',
                **spiking_kwargs
            )


        else:

            self.conv = SpikingConvEncoderLayer(
                self.embed_dim//2,
                self.embed_dim,
                kernel_size=self.first_conv_k,
                stride=2,
                padding=self.first_conv_k // 2,
                spike_norm=self.spike_norm,
                **spiking_kwargs
            )
            self.residual_encoding = spiking_residual_feature_generator(
                dim=self.embed_dim,
                norm=self.spike_norm,
                num_resblocks=self.num_res,
                cnt_fun='ADD',
                **spiking_kwargs
            )


        self.spike_proj_flag = spiking_proj
        # self.proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        # self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size[2:], stride=patch_size[2:])
        if self.spike_proj_flag:
            self.proj = SpikingEmbeddingLayer(embed_dim, embed_dim, kernel_size=3, stride=patch_size[2:], padding=1, norm = self.spike_norm, patch_resolution = self.patches_resolution, use_MS=self.use_MS, **spiking_kwargs)

        else:
            self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=patch_size[2:], padding=1)



    def forward(self, x):
        """Forward function."""
        # padding
        if x.size(1) > self.num_bins:
            x = x[:, :self.num_bins, :, :, :]

        x = x.permute(1,0,2,3,4)
        x = x.chunk(self.num_blocks, 0)
        # event_reprs = torch.stack(list(x), dim=1) # T B C H W
        # new_event_reprs = torch.zeros(event_reprs.size(0), self.num_ch, event_reprs.size(2), event_reprs.size(3),
        #                               self.num_steps).to(event_reprs.device)
        outs = []
        for i in range(self.num_blocks):
            outi = self.head(x[i])
            outi = self.conv(outi)
            outi = self.residual_encoding(outi)
            outi = self.proj(outi)
            outi = outi.unsqueeze(2)
            outs.append(outi)

        out = torch.cat(outs, dim=2)  # B, 96, 4, H, W


        # spiking_rates = {"input": torch.mean((x != 0).float())}

        # xs = self.head(x)
        # spiking_rates["head"] = torch.mean((xs != 0).float())

        # out, spiking_rates["res"] = self.residual_encoding(xs)
        # xs = self.conv(xs)
        # out = self.residual_encoding(xs)


        # out_plot = out.detach().flatten(0,1)
        # plot_2d_feature_map(out_plot[4,:,...])
        # if self.spike_proj_flag:
        #     out = self.proj(out)  # T, B, C, H, W
        #
        # else:
        #     outs = []
        #     for i in range(self.num_blocks):
        #         outi = self.proj(out[i])
        #         outi = outi.unsqueeze(2)
        #         outs.append(outi)
        #
        #     out = torch.cat(outs, dim=2)  # B, 96, B*P, H, W

            # if self.patch_norm is not None:
            #     D, Wh, Ww = out.size(2), out.size(3), out.size(4)
            #     out = out.flatten(2).transpose(1, 2)
            #     out = self.patch_norm(out)
            #     out = out.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        # out_plot = out.detach().flatten(0,1)
        # plot_2d_feature_map(out_plot[2,:,...])
        return out

    def extra_repr(self) -> str:
        return f" num_steps={self.num_steps}, patches_resolution={self.patches_resolution}"

    def flops(self):
        flops = 0
        #head
        flops += self.num_ch* self.embed_dim // 2* 3*3*self.image_size[0]*self.image_size[1]
        #bn
        flops += self.embed_dim // 2*self.image_size[0]*self.image_size[1]
        #conv
        flops += self.embed_dim // 2 *self.embed_dim * self.first_conv_k*self.first_conv_k * self.image_size[0]*self.image_size[1]//4
        #bn
        flops += self.embed_dim * self.image_size[0]*self.image_size[1]
        #residual
        flops += self.num_res * 2 * self.embed_dim * self.embed_dim * 9* self.image_size[0] * self.image_size[1]//4
        #bn
        flops += self.num_res * 2 * self.embed_dim *self.image_size[0] * self.image_size[1]//4
        #proj
        flops += self.embed_dim * self.embed_dim * 3*3 * self.patches_resolution[0] * self.patches_resolution[1]
        #bn
        flops += self.embed_dim *self.patches_resolution[0] * self.patches_resolution[1]
        return flops

    def record_flops(self):
        flops_record = {}
        #head
        flops_record["head"] = self.num_ch* self.embed_dim // 2* 3*3*self.image_size[0]*self.image_size[1]
        #bn
        # flops += self.embed_dim // 2*self.image_size[0]*self.image_size[1]
        #conv
        flops_record["conv"] =  self.embed_dim // 2 *self.embed_dim * self.first_conv_k*self.first_conv_k * self.image_size[0]*self.image_size[1]//4
        #bn
        # flops += self.embed_dim * self.image_size[0]*self.image_size[1]
        #residual
        for i in range(self.num_res):
            flops_record["res"+str(i)+"_conv0"] =  self.embed_dim * self.embed_dim * 9* self.image_size[0] * self.image_size[1]//4
            flops_record["res" + str(i)+"_conv1"] = self.embed_dim * self.embed_dim * 9 * self.image_size[0] * \
                                           self.image_size[1] // 4

        #bn
        # flops += self.num_res * 2 * self.embed_dim *self.image_size[0] * self.image_size[1]//4
        #proj
        flops_record["proj"]= self.embed_dim * self.embed_dim * 3*3 * self.patches_resolution[0] * self.patches_resolution[1]
        #bn
        # flops += self.embed_dim *self.patches_resolution[0] * self.patches_resolution[1]
        return flops_record

class Spiking_PatchEmbed_Conv_sfn(nn.Module):
    '''
    spiking patch embedding layer with 4 channel input as spike flow net
    '''
    use_MS = False
    num_res = 2
    first_conv_k = 3

    def __init__(self, img_size= (240,320),patch_size=(2, 4, 4), in_chans=10, embed_dim=96, patch_norm=None, norm = None, spiking_proj = False, spike_norm = None,  **spiking_kwargs):
        super().__init__()

        self.patch_size = patch_size
        self.image_size = img_size
        self.patches_resolution = [img_size[0] // patch_size[2] // 2, img_size[1] // patch_size[3] // 2]
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # self.num_blocks = in_chans // patch_size[0]
        self.num_bins = in_chans
        self.num_steps = spiking_kwargs['num_steps']
        self.num_ch = in_chans *2// self.num_steps

        # assert self.num_blocks ==self.num_steps

        self.spike_norm = spike_norm #norm for resnet

        self.head = SpikingConvEncoderLayer(
            self.num_ch,
            self.embed_dim // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            spike_norm=self.spike_norm,
            **spiking_kwargs
        )

        if self.use_MS:
            self.conv = MS_SpikingConvEncoderLayer(
                self.embed_dim//2,
                self.embed_dim,
                kernel_size=self.first_conv_k,
                stride=2,
                padding=self.first_conv_k // 2,
                spike_norm=self.spike_norm,
                **spiking_kwargs
            )
            self.residual_encoding = MS_spiking_residual_feature_generator(
                dim=self.embed_dim,
                norm=self.spike_norm,
                num_resblocks=self.num_res,
                cnt_fun='ADD',
                **spiking_kwargs
            )


        else:

            self.conv = SpikingConvEncoderLayer(
                self.embed_dim//2,
                self.embed_dim,
                kernel_size=self.first_conv_k,
                stride=2,
                padding=self.first_conv_k // 2,
                spike_norm=self.spike_norm,
                **spiking_kwargs
            )
            self.residual_encoding = spiking_residual_feature_generator(
                dim=self.embed_dim,
                norm=self.spike_norm,
                num_resblocks=self.num_res,
                cnt_fun='ADD',
                **spiking_kwargs
            )


        self.spike_proj_flag = spiking_proj
        # self.proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        # self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size[2:], stride=patch_size[2:])
        if self.spike_proj_flag:
            self.proj = SpikingEmbeddingLayer(embed_dim, embed_dim, kernel_size=3, stride=patch_size[2:], padding=1, norm = self.spike_norm, patch_resolution = self.patches_resolution, use_MS=self.use_MS, **spiking_kwargs)

        else:
            self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=patch_size[2:], padding=1)



    def forward(self, x):
        """Forward function."""
        # padding
        if x.size(1) > self.num_bins:
            x = x[:, :self.num_bins, :, :, :]

        event_reprs = x.permute(0, 2, 3, 4, 1)

        new_event_reprs = torch.zeros(event_reprs.size(0), self.num_ch, event_reprs.size(2), event_reprs.size(3),
                                      self.num_steps).to(event_reprs.device)

        for i in range(self.num_ch):
            start, end = i//2 * self.num_steps, (i//2 + 1) * self.num_steps
            new_event_reprs[:, i, :, :, :] = event_reprs[:, i % 2, :, :, start:end]

        x = new_event_reprs.permute(4, 0, 1, 2, 3)


        # spiking_rates = {"input": torch.mean((x != 0).float())}

        xs = self.head(x)
        # spiking_rates["head"] = torch.mean((xs != 0).float())

        # out, spiking_rates["res"] = self.residual_encoding(xs)
        xs = self.conv(xs)
        out = self.residual_encoding(xs)


        # out_plot = out.detach().flatten(0,1)
        # plot_2d_feature_map(out_plot[4,:,...])
        if self.spike_proj_flag:
            out = self.proj(out)  # T, B, C, H, W

        else:
            outs = []
            for i in range(self.num_blocks):
                outi = self.proj(out[i])
                outi = outi.unsqueeze(2)
                outs.append(outi)

            out = torch.cat(outs, dim=2)  # B, 96, B*P, H, W

            if self.patch_norm is not None:
                D, Wh, Ww = out.size(2), out.size(3), out.size(4)
                out = out.flatten(2).transpose(1, 2)
                out = self.patch_norm(out)
                out = out.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        # out_plot = out.detach().flatten(0,1)
        # plot_2d_feature_map(out_plot[2,:,...])
        return out

    def extra_repr(self) -> str:
        return f" num_steps={self.num_steps}, patches_resolution={self.patches_resolution}"

    def flops(self):
        flops = 0
        #head
        flops += self.num_ch* self.embed_dim // 2* 3*3*self.image_size[0]*self.image_size[1]
        #bn
        flops += self.embed_dim // 2*self.image_size[0]*self.image_size[1]
        #conv
        flops += self.embed_dim // 2 *self.embed_dim * self.first_conv_k*self.first_conv_k * self.image_size[0]*self.image_size[1]//4
        #bn
        flops += self.embed_dim * self.image_size[0]*self.image_size[1]
        #residual
        flops += self.num_res * 2 * self.embed_dim * self.embed_dim * 9* self.image_size[0] * self.image_size[1]//4
        #bn
        flops += self.num_res * 2 * self.embed_dim *self.image_size[0] * self.image_size[1]//4
        #proj
        flops += self.embed_dim * self.embed_dim * 3*3 * self.patches_resolution[0] * self.patches_resolution[1]
        #bn
        flops += self.embed_dim *self.patches_resolution[0] * self.patches_resolution[1]
        return flops

    def record_flops(self):
        flops_record = {}
        #head
        flops_record["head"] = self.num_ch* self.embed_dim // 2* 3*3*self.image_size[0]*self.image_size[1]
        #bn
        # flops += self.embed_dim // 2*self.image_size[0]*self.image_size[1]
        #conv
        flops_record["conv"] =  self.embed_dim // 2 *self.embed_dim * self.first_conv_k*self.first_conv_k * self.image_size[0]*self.image_size[1]//4
        #bn
        # flops += self.embed_dim * self.image_size[0]*self.image_size[1]
        #residual
        for i in range(self.num_res):
            flops_record["res"+str(i)+"_conv0"] =  self.embed_dim * self.embed_dim * 9* self.image_size[0] * self.image_size[1]//4
            flops_record["res" + str(i)+"_conv1"] = self.embed_dim * self.embed_dim * 9 * self.image_size[0] * \
                                           self.image_size[1] // 4

        #bn
        # flops += self.num_res * 2 * self.embed_dim *self.image_size[0] * self.image_size[1]//4
        #proj
        flops_record["proj"]= self.embed_dim * self.embed_dim * 3*3 * self.patches_resolution[0] * self.patches_resolution[1]
        #bn
        # flops += self.embed_dim *self.patches_resolution[0] * self.patches_resolution[1]
        return flops_record

class MS_Spiking_PatchEmbed_Conv_sfn(Spiking_PatchEmbed_Conv_sfn):
    use_MS = True

class MS_PED_Spiking_PatchEmbed_Conv_sfn(nn.Module):
    '''
    spiking patch embedding layer with PED with 4 channel input as spike flow net
    MS shortcut
    '''
    use_MS = True
    num_res = 2
    first_conv_k = 3

    def __init__(self, img_size= (240,320),patch_size=(2, 4, 4), in_chans=10, embed_dim=96, patch_norm=None, norm = None, spiking_proj = False, spike_norm = None,  **spiking_kwargs):
        super().__init__()

        self.patch_size = patch_size
        self.image_size = img_size
        self.patches_resolution = [img_size[0] // patch_size[2] // 2, img_size[1] // patch_size[3] // 2]
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # self.num_blocks = in_chans // patch_size[0]
        self.num_bins = in_chans
        self.num_steps = spiking_kwargs['num_steps']
        self.num_ch = in_chans *2// self.num_steps

        # assert self.num_blocks ==self.num_steps

        self.spike_norm = spike_norm #norm for resnet

        self.head = SpikingConvEncoderLayer(
            self.num_ch,
            self.embed_dim // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            spike_norm=self.spike_norm,
            **spiking_kwargs
        )


        self.conv = MS_SpikingConvEncoderLayer(
            self.embed_dim//2,
            self.embed_dim,
            kernel_size=self.first_conv_k,
            stride=2,
            padding=self.first_conv_k // 2,
            spike_norm=self.spike_norm,
            **spiking_kwargs
        )
        self.residual_encoding = MS_spiking_residual_feature_generator(
            dim=self.embed_dim,
            norm=self.spike_norm,
            num_resblocks=self.num_res,
            cnt_fun='ADD',
            **spiking_kwargs
        )


        self.proj = SpikingPEDLayer(embed_dim, embed_dim, kernel_size=3, stride=patch_size[2:], padding=1, norm = self.spike_norm, patch_resolution = self.patches_resolution, **spiking_kwargs)




    def forward(self, x):
        """Forward function."""
        if x.size(1) > self.num_bins:
            x = x[:, :self.num_bins, :, :, :]

        event_reprs = x.permute(0, 2, 3, 4, 1)

        new_event_reprs = torch.zeros(event_reprs.size(0), self.num_ch, event_reprs.size(2), event_reprs.size(3),
                                      self.num_steps).to(event_reprs.device)

        for i in range(self.num_ch):
            start, end = i//2 * self.num_steps, (i//2 + 1) * self.num_steps
            new_event_reprs[:, i, :, :, :] = event_reprs[:, i % 2, :, :, start:end]

        x = new_event_reprs.permute(4, 0, 1, 2, 3)

        xs = self.head(x)
        xs = self.conv(xs)
        out = self.residual_encoding(xs)
        out = self.proj(out)  # T, B, C, H, W
        return out

    def extra_repr(self) -> str:
        return f" num_steps={self.num_steps}, patches_resolution={self.patches_resolution}"

    def flops(self):
        flops = 0
        #head
        flops += self.num_ch* self.embed_dim // 2* 3*3*self.image_size[0]*self.image_size[1]
        #bn
        flops += self.embed_dim // 2*self.image_size[0]*self.image_size[1]
        #conv
        flops += self.embed_dim // 2 *self.embed_dim * self.first_conv_k*self.first_conv_k * self.image_size[0]*self.image_size[1]//4
        #bn
        flops += self.embed_dim * self.image_size[0]*self.image_size[1]
        #residual
        flops += self.num_res * 2 * self.embed_dim * self.embed_dim * 9* self.image_size[0] * self.image_size[1]//4
        #bn
        flops += self.num_res * 2 * self.embed_dim *self.image_size[0] * self.image_size[1]//4
        #proj
        flops += self.embed_dim * self.embed_dim * 3*3 * self.patches_resolution[0] * self.patches_resolution[1]
        #bn
        flops += self.embed_dim *self.patches_resolution[0] * self.patches_resolution[1]
        return flops

    def record_flops(self):
        flops_record = {}
        #head
        flops_record["head"] = self.num_ch* self.embed_dim // 2* 3*3*self.image_size[0]*self.image_size[1]
        #bn
        # flops += self.embed_dim // 2*self.image_size[0]*self.image_size[1]
        #conv
        flops_record["conv"] =  self.embed_dim // 2 *self.embed_dim * self.first_conv_k*self.first_conv_k * self.image_size[0]*self.image_size[1]//4
        #bn
        # flops += self.embed_dim * self.image_size[0]*self.image_size[1]
        #residual
        for i in range(self.num_res):
            flops_record["res"+str(i)+"_conv0"] =  self.embed_dim * self.embed_dim * 9* self.image_size[0] * self.image_size[1]//4
            flops_record["res" + str(i)+"_conv1"] = self.embed_dim * self.embed_dim * 9 * self.image_size[0] * \
                                           self.image_size[1] // 4

        #bn
        # flops += self.num_res * 2 * self.embed_dim *self.image_size[0] * self.image_size[1]//4
        #proj
        flops_record["proj"]= self.embed_dim * self.embed_dim * 3*3 * self.patches_resolution[0] * self.patches_resolution[1]
        #bn
        # flops += self.embed_dim *self.patches_resolution[0] * self.patches_resolution[1]
        return flops_record




if __name__ == "__main__":
    _seed_ = 16146
    random.seed(_seed_)
    torch.manual_seed(_seed_)
    torch.cuda.manual_seed_all(_seed_)
    # (batch_size, num_bins, num_polarities, height, width)
    chunk = torch.rand([1, 10, 2, 240, 320])  # ( B, bin, C, H, W)
    spiking_kwargs = {"num_steps": 5, "v_reset": None, "v_th": 0.5,"neuron_type": "lif", "surrogate_fun": 'surrogate.ATan()', "tau": 2., "detach_reset": True, "spike_norm": "BN"}
    # model = MS_Spiking_PatchEmbed_Conv_sfn(img_size= (240,320),patch_size=(1, 1, 2, 2), in_chans=10, embed_dim=96,spiking_proj=True,**spiking_kwargs)
    model = MS_PED_Spiking_PatchEmbed_Conv_sfn(img_size=(240, 320), patch_size=(1, 1, 2, 2), in_chans=10, embed_dim=96,
                                           spiking_proj=True, **spiking_kwargs)
    functional.reset_net(model)
    functional.set_step_mode(model, "m")
    print('--- Test Model ---')
    print(model)
    outps = model(chunk)