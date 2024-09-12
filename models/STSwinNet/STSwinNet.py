import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from models.model_util import CropSize,copy_states,CropParameters
from models.base import BaseModel
from models.unet import BaseUNet
from .STswin_transformer import SwinTransformer3D
from .swin_transformer3D_v2 import SwinTransformer3D_v2
from models.submodules import ResidualBlock,ConvLayer,UpsampleConvLayer,TransposedConvLayer
from timm.models.layers import DropPath, trunc_normal_


class STT_encoder(nn.Module):
    v2=True
    def __init__(
            self,
            arc_type = "swinv2",
            patch_embed_type = "PatchEmbedLocal",
            img_size=(240,320),
            patch_size=(32, 2, 2),
            in_chans=128,
            embed_dim=96,
            depths=[2, 2, 6],
            num_heads=[3, 6, 12],
            window_size=[2, 7, 7],
            pretrained_window_size=[0, 0, 0],
            mlp_ratio=4.,
            patch_norm=False,
            out_indices=(0, 1, 2),
            frozen_stages=-1,
            pol_in_channel = False,
            norm = None,
            **spiking_kwargs
    ):
        super(STT_encoder, self).__init__()
        #
        self.num_blocks = in_chans // patch_size[0]+1
        if self.v2:
            self.num_blocks = in_chans // patch_size[0]
        if pol_in_channel:
            self.num_blocks = self.num_blocks * 2
        # self.out_num_depths = self.num_blocks - 1
        # self.block_channel = block_channel
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.num_encoders = len(self.depths)
        self.out_channels = [self.embed_dim * (2 ** i) for i in range(self.num_encoders)]

        if arc_type == "swinv1":

            self.swin3d = SwinTransformer3D(
                embed_type = patch_embed_type,
                patch_size=self.patch_size,
                in_chans=self.in_chans,
                embed_dim=self.embed_dim,
                depths=self.depths,
                num_heads=self.num_heads,
                window_size=self.window_size,
                pretrained_window_size=pretrained_window_size,
                mlp_ratio=self.mlp_ratio,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=self.patch_norm,
                norm = norm,
                out_indices=self.out_indices,
                frozen_stages=self.frozen_stages,
                **spiking_kwargs

            )
        elif arc_type == "swinv2":
            self.swin3d = SwinTransformer3D_v2(
                embed_type=patch_embed_type,
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_chans=self.in_chans,
                embed_dim=self.embed_dim,
                depths=self.depths,
                num_heads=self.num_heads,
                window_size=self.window_size,
                pretrained_window_size=pretrained_window_size,
                mlp_ratio=self.mlp_ratio,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=self.patch_norm,
                norm=norm,
                out_indices=self.out_indices,
                frozen_stages=self.frozen_stages,
                **spiking_kwargs
            )

        # self.patches_T = self.num_blocks
        # self.patch_T = self.patches_T // self.num_blocks  # 1
        self.projections = self.build_projections()


    def build_projections(self):
        conv_layers = nn.ModuleList()
        for i in range(self.num_encoders):
            conv_layer_i = nn.ModuleList()
            for ti in range(self.num_blocks): #time steps
                conv_layer_i.append(nn.Conv2d(self.out_channels[i], self.out_channels[i] // self.num_blocks, 1))
            conv_layers.append(conv_layer_i)
        return conv_layers


    def forward(self, inputs):
        features, spiking_rate = self.swin3d(inputs)

        outs = []

        #concatenate encoder features along temporal bins and project to B,C,H,W
        for i in range(self.num_encoders): #swin number
            out_layer_i = []
            features_i = features[i].chunk(self.num_blocks, 2)
            #print(features_i[0].shape)
            B, C, T, H, W = features_i[0].shape
            # features_i = features_i.reshape(B, -1, H, W)
            for k in range(self.num_blocks):
                feature_k = features_i[k].reshape(B, -1, H, W)  # B,C,H,W
                out_k = self.projections[i][k](feature_k)
                out_layer_i.append(out_k)
            out_i = torch.cat(out_layer_i, dim=1) #for each encoder cat 4 blocks featuremaps
            outs.append(out_i) #C W/2 H/2   2C W/4 H/4 4c ...


        return outs, spiking_rate

class STT_MultiResUNet(BaseUNet):
    """
    UNet architecture with swin transformer encoder
    Symmetric, skip connections on every encoding layer.
    Predictions at each decoding layer.
    Predictions are added as skip connection (concat) to the input of the subsequent layer.

    """
    pol_channel = False
    encoder_block = STT_encoder
    ff_type = ConvLayer
    res_type = ResidualBlock
    upsample_type = UpsampleConvLayer
    transpose_type = TransposedConvLayer
    w_scale_pred = None
    def __init__(self, unet_kwargs, stt_kwargs):
        self.final_activation = unet_kwargs.pop("final_activation", None)
        super().__init__(**unet_kwargs)

        self.arc_type = stt_kwargs["use_arc"][0]
        self.patch_embed_type = stt_kwargs["use_arc"][1]
        self.num_bins_events = unet_kwargs['num_bins']
        self.depths=[int(i) for i in stt_kwargs["swin_depths"]]
        self.num_heads=[int(i) for i in stt_kwargs["swin_num_heads"]]
        assert len(self.depths) == self.num_encoders
        assert len(self.num_heads) == self.num_encoders
        self.patch_size=[int(i) for i in stt_kwargs["swin_patch_size"]]
        self.out_indices=[int(i) for i in stt_kwargs["swin_out_indices"]]
        self.window_size = [int(i) for i in stt_kwargs["window_size"]]
        self.pretrained_window_size =  [int(i) for i in stt_kwargs["pretrained_window_size"]]
        self.mlp_ratio = stt_kwargs["mlp_ratio"]
        self.input_size = stt_kwargs["input_size"]


        self.encoder_output_sizes = [
            int(self.base_num_channels * pow(self.channel_multiplier, i)) for i in range(self.num_encoders)
        ]
        self.encoder_input_sizes = self.encoder_output_sizes.copy()
        self.encoder_input_sizes.insert(0,self.base_num_channels)
        self.encoder_input_sizes.pop()
        self.max_num_channels = self.encoder_output_sizes[-1]

        #self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders-1)

        # self.num_channel_spikes = config["num_channel_spikes"]


        #print('----- ', self.num_heads)

        self.encoders = self.encoder_block(
            arc_type=self.arc_type,
            patch_embed_type=self.patch_embed_type,
            img_size=self.input_size,
            patch_size=self.patch_size,
            in_chans=self.num_bins_events,
            embed_dim=self.base_num_channels,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            pretrained_window_size=self.pretrained_window_size,
            mlp_ratio=self.mlp_ratio,
            out_indices=self.out_indices,
            norm=self.norm,
            pol_in_channel= self.pol_channel,
            **self.spiking_kwargs
        )
        self.resblocks = self.build_resblocks()
        self.decoders = self.build_multires_prediction_decoders()
        self.preds = self.build_multires_prediction_layer()


    def build_resblocks(self):
        resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            resblocks.append(
                self.res_type(
                    self.max_num_channels,
                    self.max_num_channels,
                    activation=self.ff_act,
                    norm=self.norm
                )
            )
        return resblocks
    def build_encoders(self):
        pass

    def build_multires_prediction_layer(self):
        preds = nn.ModuleList()
        # decoder_output_sizes_r = [int(self.base_num_channels * pow(self.channel_multiplier, i - 1)) for i in range(self.num_encoders)]

        # decoder_output_sizes = reversed(decoder_output_sizes_r)
        decoder_output_sizes = reversed(self.encoder_input_sizes)

        for output_size in decoder_output_sizes:
            preds.append(
                self.ff_type(output_size, self.num_output_channels, 1, activation=self.final_activation, norm=None)
            )
        return preds

    def build_multires_prediction_decoders(self):
        decoders = nn.ModuleList()
        #decoder_output_sizes_r = [int(self.base_num_channels * pow(self.channel_multiplier, i - 1)) for i in range(self.num_encoders)]
        #decoder_output_sizes = reversed(decoder_output_sizes_r)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            prediction_channels = 0 if i == 0 else self.num_output_channels
            decoders.append(
                self.UpsampleLayer(
                    2 * input_size + prediction_channels,
                    output_size,
                    kernel_size=self.kernel_size,
                    activation=self.ff_act,
                    norm=self.norm
                )
            )
        return decoders


    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """
        # encoder


        block, spiking_rate = self.encoders(x)


        x = block[-1]
        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder and multires predictions
        predictions = []
        for i, (decoder, pred) in enumerate(zip(self.decoders, self.preds)):
            x = self.skip_ftn(x, block[self.num_encoders - i - 1])
            if i > 0:
                x = self.skip_ftn(predictions[-1], x)
            x = decoder(x)
            predictions.append(pred(x))
        return predictions, spiking_rate

    def flops(self):
        flops = 0
        #encoder
        flops += self.encoders.swin3d.flops()
        H, W = self.encoders.swin3d.patch_embed.patches_resolution #96 72
        # size for the last encoder output
        H = H // 2 ** (self.num_encoders-1)
        W = W // 2 ** (self.num_encoders-1)
        #residual blocks 2 convs in each
        flops += 2* self.max_num_channels * self.max_num_channels *3 *3  * H * W *self.num_residual_blocks
        #decoder and multires predictions
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        for i, (input_size, output_size) in enumerate(zip(decoder_input_sizes, decoder_output_sizes)):
            prediction_channels = 0 if i == 0 else self.num_output_channels
            H = H * 2
            W = W * 2
            flops += (2 * input_size + prediction_channels) * output_size * H * W * self.kernel_size * self.kernel_size
            flops += output_size * self.num_output_channels * H * W


        return flops


class STTFlowNet(BaseModel):
    """

    3 encoders

    encoder: convlstm
    decoder

    """
    unet_type = STT_MultiResUNet
    recurrent_block_type = "none"
    spiking_feedforward_block_type = None
    num_en = 3

    def __init__(self, unet_kwargs, stt_kwargs):
        super().__init__()

        norm = None
        use_upsample_conv = True
        if "norm" in unet_kwargs.keys():
            norm = unet_kwargs["norm"]
        if "use_upsample_conv" in unet_kwargs.keys():
            use_upsample_conv = unet_kwargs["use_upsample_conv"]
        STTFlowNet_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": self.num_en,
            "num_residual_blocks": 2,
            "num_output_channels": 2,
            "skip_type": "concat",
            "norm": norm,
            "use_upsample_conv": use_upsample_conv,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "recurrent_block_type": self.recurrent_block_type,
            "final_activation": unet_kwargs["final_activation"],
            "spiking_feedforward_block_type": self.spiking_feedforward_block_type,
            "spiking_neuron": unet_kwargs["spiking_neuron"],

        }

        self.crop = None
        self.mask = unet_kwargs["mask_output"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.encoding = unet_kwargs["encoding"]
        self.num_bins = unet_kwargs["num_bins"]
        self.num_encoders = STTFlowNet_kwargs["num_encoders"]
        self.num_split = self.num_bins  // stt_kwargs["swin_patch_size"][0]
        self.final_activation = STTFlowNet_kwargs["final_activation"]


        unet_kwargs.update(STTFlowNet_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("encoding", None)
        unet_kwargs.pop("round_encoding", None)
        unet_kwargs.pop("norm_input", None)
        unet_kwargs.pop("mask_output", None)

        self.sttmultires_unet = self.unet_type(unet_kwargs, stt_kwargs)



    def detach_states(self):
        pass

    def reset_states(self):
        pass
    def normalize(self,x):

        mean, stddev = (
            x[x != 0].mean(),
            x[x != 0].std(),
        )
        if stddev > 0:
            x[x != 0] = (x[x != 0] - mean) / stddev
        return x
    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

        self.apply(_init_weights)




    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_tensor: N x num_bins x H x W   normalized t (-1 1)

        :param event_cnt: N x 4 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # input encoding
        if self.encoding == "voxel":
            x = event_voxel

        elif self.encoding == "cnt":
            x = event_cnt
        else:
            print("Model error: Incorrect input encoding.")
            raise AttributeError

        if x.size(1) != self.num_bins:
            chunk1 = x[:, :self.num_bins, :, :]
            chunk2 = x[:, self.num_bins:, :, :]
            if self.norm_input:
                chunk1 = self.normalize(chunk1)
                chunk2 = self.normalize(chunk2)
            # Feature maps [f_0 :: f_i :: f_g]
            inputs = chunk2.chunk(self.num_split, dim=1)
            inputref = chunk1.chunk(self.num_split, dim=1)[-1]
            x = (inputref,) + inputs  # [group+1] elements
            x = torch.stack(list(x), dim=1).permute(1, 0, 2, 3, 4)  # T B C H W
        else:
            x = x.chunk(self.num_split, dim=1)
            x = torch.stack(list(x), dim=1).permute(1, 0, 2, 3, 4)



        H = x.size(-2)
        W = x.size(-1)
        # x = torch.transpose(x, 1, 2)
        # pad size for input
        factor = {'h':2, 'w':2} # patch size for l0
        pad_crop = CropSize(W, H, factor)
        if (H % factor['h'] != 0) or (W % factor['w'] != 0):
            x = pad_crop.pad(x)

        multires_flow,spiking_rate = self.sttmultires_unet.forward(x)

        # log att
        if log:
            attns = self.sttmultires_unet.encoders.swin3d.get_layer_attention_scores(x)
        else:
            attns = None


        flow_list = []
        #interpolate the final flow
        #flow_h = multires_flow[-1].shape[2]
        #flow_w = multires_flow[-1].shape[3]
        flow_h = H
        flow_w = W
        for flow in multires_flow:
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow,
                    scale_factor=(
                        flow_h / flow.shape[2],
                        flow_w / flow.shape[3],
                    ),
                )
            )

        # crop output
        if self.crop is not None:
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                flow_list[i] = flow_list[i].contiguous()




        return {"flow": flow_list, "attn": attns, "spiking_rates":spiking_rate}

    def flops(self):
        return self.sttmultires_unet.flops()

class STTFlowNet_4en(STTFlowNet):
    """

    3 encoders

    encoder: convlstm
    decoder

    """
    unet_type = STT_MultiResUNet
    recurrent_block_type = "none"
    spiking_feedforward_block_type = None
    num_en = 4

