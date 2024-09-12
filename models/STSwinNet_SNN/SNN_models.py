"""
implement with spiking jelly
"""
import torch
from ..base import BaseModel
import torch.nn as nn
from ..model_util import *
from .Spiking_modules import *
from ..submodules import ConvLayer


class SpikingMultiResUNet(nn.Module):
    """
    Spiking UNet architecture with SEW shortcut.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """


    ff_type = SpikingConvEncoderLayer
    res_type = SEWResBlock
    upsample_type = SpikingDecoderLayer
    transpose_type = SpikingTransposeDecoderLayer
    pred_type = SpikingPredLayer
    input_sfn = True
    w_scale_pred = 0.01
    upsample_4 = False

    def __init__(
        self,
        base_num_channels,
        num_encoders,
        num_residual_blocks,
        num_output_channels,
        skip_type,
        norm,
        use_upsample_conv,
        num_bins,
        recurrent_block_type=None,
        kernel_size=5,
        channel_multiplier=2,
        activations=["relu", None],
        final_activation = None,
        spiking_neuron=None,
    ):
        super(SpikingMultiResUNet, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = None #norm for ANN
        self.recurrent_block_type = recurrent_block_type
        self.channel_multiplier = channel_multiplier
        self.ff_act, self.rec_act = activations
        self.final_activation = final_activation

        self.num_bins_all = num_bins

        self.spiking_kwargs = {}
        if type(spiking_neuron) is dict:
            self.spiking_kwargs.update(spiking_neuron)
            self.steps = self.spiking_kwargs["num_steps"]
            self.num_ch = num_bins * 2 // self.steps

        self.skip_ftn = eval("skip_" + skip_type)
        if use_upsample_conv:
            self.UpsampleLayer = self.upsample_type
        else:
            self.UpsampleLayer = self.transpose_type
        assert self.num_output_channels > 0

        self.encoder_input_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i)) for i in range(self.num_encoders)
        ]

        self.encoder_output_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i + 1)) for i in range(self.num_encoders)
        ]
        self.max_num_channels = self.encoder_output_sizes[-1]
        #
        # self.final_activation = unet_kwargs.pop("final_activation", None)

        # self.head = self.ff_type(
        #             self.num_bins,
        #             self.base_num_channels,
        #             kernel_size=kernel_size,
        #             stride=1,
        #             padding=1,
        #             **self.spiking_kwargs
        #         )
        self.encoders = self.build_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_multires_prediction_decoders()
        self.preds = self.build_multires_prediction_layer()



    def build_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_ch
            encoders.append(
                self.ff_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.kernel_size//2,
                    **self.spiking_kwargs
                )
            )
        return encoders

    def build_resblocks(self):
        resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            resblocks.append(
                self.res_type(
                    self.max_num_channels,
                    self.max_num_channels,
                    connect_function='ADD',
                    **self.spiking_kwargs
                )
            )
        return resblocks

    def build_multires_prediction_layer(self):
        preds = nn.ModuleList()
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        for output_size in decoder_output_sizes:
            preds.append(
                self.pred_type(
                    output_size,
                    self.num_output_channels,
                    1,
                    **self.spiking_kwargs
                )
            )
        return preds

    def build_multires_prediction_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        i_max = len(self.encoder_input_sizes)-1
        sf = 2
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            prediction_channels = 0 if i == 0 else self.num_output_channels
            if self.upsample_4:
                sf = 4 if i == i_max else 2
            decoders.append(
                self.UpsampleLayer(
                    2 * input_size + prediction_channels,
                    output_size,
                    kernel_size=self.kernel_size,
                    scale = sf,
                    **self.spiking_kwargs
                )
            )
        return decoders

    def forward(self, x):
        # encoder
        blocks = []

        # sfn
        if x.size(1) > self.num_bins_all:
            x = x[:, :self.num_bins_all, :, :, :]


        # B, T, C, H, W = x.size()
        if self.input_sfn:
            event_reprs = x.permute(0, 2, 3, 4, 1)

            new_event_reprs = torch.zeros(event_reprs.size(0), self.num_ch, event_reprs.size(2), event_reprs.size(3),
                                          self.steps).to(event_reprs.device)

            for i in range(self.num_ch):
                start, end = i//2 * self.steps, (i//2 + 1) * self.steps
                new_event_reprs[:, i, :, :, :] = event_reprs[:, i % 2, :, :, start:end]

            x = new_event_reprs.permute(4, 0, 1, 2, 3)
        else:

            x = x.view([x.shape[0], -1] + list(x.shape[3:]))
            xs = x.chunk(self.steps, 1)
            x = torch.stack(list(xs), dim=1).permute(1, 0, 2, 3, 4)  #



        # x = self.head(x)
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            blocks.append(x)

        # residual blocks

        for i, resblock in enumerate(self.resblocks):
            x = resblock(x)

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_ftn(x, blocks[self.num_encoders - i - 1],dim=2)
            if i > 0:
                x = self.skip_ftn(predictions[-1], x,dim=2)
            x = decoder(x)
            pred_out = pred(x)
            predictions.append(pred_out)


        return predictions

