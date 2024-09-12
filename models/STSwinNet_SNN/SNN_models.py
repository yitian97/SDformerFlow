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
    Spiking recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    """

    """
     Spiking recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
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
        """
        :param x: N x num_input_channels x H x W
        :return: [N x num_output_channels x H x W for i in range(self.num_encoders)]
        """

        # encoder
        blocks = []
        # spiking_rates = {"input": torch.mean((x != 0).float())}

        # out_plot = x.detach()
        # plot_2d_feature_map(out_plot[0, :,...])
        #if polarity not in channel:
        # xs = x.chunk(self.steps, 1)
        # x = torch.stack(list(xs), dim=1).permute(1, 0, 2, 3, 4)  #
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
            # spiking_rates["encoder"+str(i)]=torch.mean((x != 0).float())
            blocks.append(x)

        # residual blocks

        for i, resblock in enumerate(self.resblocks):
            x = resblock(x)
            # spiking_rates["res" + str(i)] = torch.mean((x != 0).float())

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_ftn(x, blocks[self.num_encoders - i - 1],dim=2)
            if i > 0:
                x = self.skip_ftn(predictions[-1], x,dim=2)
            x = decoder(x)
            # spiking_rates["decoder" + str(i)] = torch.mean((x != 0).float())

            # pred_out = []
            #
            # for i in range(self.steps):
            #     pred_i = pred(x[i])
            #     pred_i = pred_i.unsqueeze(0)
            #     pred_out.append(pred_i)
            # pred_out = torch.cat(pred_out,dim=0)
            pred_out = pred(x)
            predictions.append(pred_out)


        return predictions

class MS_SpikingMultiResUNet(nn.Module):
    """
    Spiking recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.
    Memory short cut
    """

    """
     Spiking recurrent UNet architecture where every encoder is followed by a recurrent convolutional block.
     Symmetric, skip connections on every encoding layer.
     Predictions at each decoding layer.
     Predictions are added as skip connection (concat) to the input of the subsequent layer.
     """

    ff_type = MS_SpikingConvEncoderLayer
    res_type = MS_ResBlock
    upsample_type = MS_SpikingDecoderLayer
    pred_type = ConvLayer
    w_scale_pred = 0.01

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
        super(MS_SpikingMultiResUNet, self).__init__()
        self.base_num_channels = base_num_channels
        self.num_encoders = num_encoders
        self.num_residual_blocks = num_residual_blocks
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.skip_type = skip_type
        self.norm = norm
        self.recurrent_block_type = recurrent_block_type
        self.channel_multiplier = channel_multiplier
        self.ff_act, self.rec_act = activations
        self.final_activation = final_activation

        self.num_bins = num_bins

        self.spiking_kwargs = {}
        if type(spiking_neuron) is dict:
            self.spiking_kwargs.update(spiking_neuron)
            self.steps = self.spiking_kwargs["num_steps"]
            self.num_bins =num_bins //self.spiking_kwargs["num_steps"]

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


        self.encoders = self.build_encoders()
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_multires_prediction_decoders()
        self.preds = self.build_multires_prediction_layer()


    def build_encoders(self):
        encoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(self.encoder_input_sizes, self.encoder_output_sizes)):
            if i == 0:
                input_size = self.num_bins
            encoders.append(
                self.ff_type(
                    input_size,
                    output_size,
                    kernel_size=self.kernel_size,
                    stride=2,
                    **self.spiking_kwargs
                )
            )
        encoders[0].sn = None
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
                    activation=self.final_activation,
                    norm=None,
                    w_scale=self.w_scale_pred,
                )
            )
        return preds
    def build_multires_prediction_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            prediction_channels = 0 if i == 0 else self.num_output_channels
            decoders.append(
                self.UpsampleLayer(
                    2 * input_size + prediction_channels,
                    output_size,
                    kernel_size=self.kernel_size,
                    **self.spiking_kwargs
                )
            )
        return decoders
    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: [N x num_output_channels x H x W for i in range(self.num_encoders)]
        """

        # encoder
        blocks = []
        # spiking_rates = {"input": torch.mean((x != 0).float())}

        # out_plot = x.detach()
        # plot_2d_feature_map(out_plot[0, :,...])
        xs = x.chunk(self.steps, 1)
        x = torch.stack(list(xs), dim=1).permute(1, 0, 2, 3, 4)  #



        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            # spiking_rates["encoder"+str(i)]=torch.mean((x != 0).float())
            blocks.append(x)

        # residual blocks

        for i, resblock in enumerate(self.resblocks):
            x = resblock(x)
            # spiking_rates["res" + str(i)] = torch.mean((x != 0).float())

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_ftn(x, blocks[self.num_encoders - i - 1],dim=2)
            if i > 0:
                x = self.skip_ftn(predictions[-1], x,dim=2)
            x = decoder(x)
            # spiking_rates["decoder" + str(i)] = torch.mean((x != 0).float())

            pred_out = []

            for i in range(self.steps):
                pred_i = pred(x[i])
                pred_i = pred_i.unsqueeze(0)
                pred_out.append(pred_i)
            pred_out = torch.cat(pred_out,dim=0)
            predictions.append(pred_out)


        return predictions

class SpikingEVFlowNet(nn.Module):
    """
    Spiking version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    Implemented using spikingjelly
    """

    unet_type = SpikingMultiResUNet
    recurrent_block_type = None



    def __init__(self, unet_kwargs):
        super().__init__()

        norm = None
        use_upsample_conv = True
        if "norm" in unet_kwargs.keys():
            norm = unet_kwargs["norm"]
        if "use_upsample_conv" in unet_kwargs.keys():
            use_upsample_conv = unet_kwargs["use_upsample_conv"]
        if "num_steps" in unet_kwargs.keys():
            unet_kwargs["spiking_neuron"]["num_steps"] = unet_kwargs["num_steps"]
            unet_kwargs.pop("num_steps", None)

        FlowNet_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 4,
            "num_residual_blocks": 2,
            "num_output_channels": 2,
            "skip_type": "concat",
            "norm": norm,
            "use_upsample_conv": use_upsample_conv,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "recurrent_block_type": self.recurrent_block_type,
            "final_activation":  unet_kwargs["final_activation"],
            "spiking_neuron": unet_kwargs["spiking_neuron"],
        }

        self.crop = None
        self.mask = unet_kwargs["mask_output"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.encoding = unet_kwargs["encoding"]
        self.num_bins = unet_kwargs["num_bins"]
        self.num_encoders = FlowNet_kwargs["num_encoders"]

        unet_kwargs.update(FlowNet_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("encoding", None)
        unet_kwargs.pop("round_encoding", None)
        unet_kwargs.pop("norm_input", None)
        unet_kwargs.pop("mask_output", None)

        self.multires_unetrec = self.unet_type(**unet_kwargs)

        self.pooling = neuron.IFNode(v_threshold=float('inf'), v_reset=0.)


    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, x, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # pad input
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass
        multires_flow = self.multires_unetrec.forward(x)

        # upsample flow estimates to the original input resolution
        flow_list = []

        # for flow_res in multires_flow:
        #     # flow = torch.sum(flow, dim = 0)
        #     flow = []
        #     steps = multires_flow[0].shape[0]
        #     for i in range(steps):
        #         flow_i = torch.nn.functional.interpolate(
        #             flow_res[i],
        #             scale_factor=(
        #                 multires_flow[-1].shape[-2] / flow_res[i].shape[-2],
        #                 multires_flow[-1].shape[-1] / flow_res[i].shape[-1],
        #             ),
        #         )
        #
        #         flow.append(flow_i.unsqueeze(0))
        #     flow = torch.cat(flow, dim=0)
        #     self.pooling(flow)
        #     flow = self.pooling.v
        #     flow_list.append(flow)

        for flow in multires_flow:
            flow = torch.sum(flow, dim = 0)
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow,
                    scale_factor=(
                        multires_flow[-1].shape[-2] / flow.shape[-2],
                        multires_flow[-1].shape[-1] / flow.shape[-1],
                    ),
                )
            )


        # crop output
        if self.crop is not None:
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                flow_list[i] = flow_list[i].contiguous()

        return {"flow": flow_list}

class SpikingEVFlowNet_v2(SpikingEVFlowNet):
    """
    Spiking version of the EV-FlowNet architecture from the paper "EV-FlowNet: Self-Supervised Optical
    Flow for Event-based Cameras", Zhu et al., RSS 2018.
    Implemented using spikingjelly
    """

    unet_type = MS_SpikingMultiResUNet



