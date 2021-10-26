#!/bin/python
# -----------------------------------------------------------------------------
# File Name : fpn_decolle.py
# Author: Sami BARCHID
#
# Copyright : (c) CRIStAL-FoX, Sami BARCHID
# Licence : GPLv2
# -----------------------------------------------------------------------------
from snn.base import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FpnLocalization(DECOLLEBase):
    def __init__(
        self,
        in_channels,
        nb_classes,
        alpha=0.9,
        beta=0.85,
        alpharp=0.65,
        lc_ampl=0.5,
        deltat=1000,
        input_shape=(88, 120)
    ):
        super(FpnLocalization, self).__init__()
        # index in layer list where the decoder module starts (0 for forcing bbox regression everywhere)
        self.decoder_start = 0
        self.nb_classes = nb_classes
        self.poolings = nn.ModuleList()  # adds ModuleList composed of Downsampling or Upsampling layers

        # ENCODER INITIALIZATION
        ###################################
        # Encoder - Block 1
        self._LIFConvBlock(
            in_channels,
            32,
            linear_readout_shape=(input_shape[0]/2, input_shape[1]/2),
            kernel_size=5,
            alpha=alpha,
            beta=beta,
            alpharp=alpharp,
            deltat=deltat,
            pool_size=2,
            pool_method="down",
            lc_ampl=lc_ampl,
            # readout_channels=nb_classes
            # 4 channels for readout = (coordinates for bbox)
            readout_channels=4,
        )

        # Encoder - Block 2
        self._LIFConvBlock(
            32,
            64,
            kernel_size=3,
            alpha=alpha,
            beta=beta,
            alpharp=alpharp,
            deltat=deltat,
            pool_size=2,
            pool_method="down",
            lc_ampl=lc_ampl,
            # readout_channels=nb_classes,
            # 4 channels for readout = (coordinates for bbox)
            readout_channels=4,
            linear_readout_shape=(input_shape[0]/4, input_shape[1]/4)
        )

        # Encoder - Block 3
        self._LIFConvBlock(
            64,
            128,
            kernel_size=3,
            alpha=alpha,
            beta=beta,
            alpharp=alpharp,
            deltat=deltat,
            pool_size=2,
            pool_method="none",
            lc_ampl=lc_ampl,
            # readout_channels=nb_classes,
            # 4 channels for readout = (coordinates for bbox)
            readout_channels=4,
            linear_readout_shape=(input_shape[0]/4, input_shape[1]/4)
        )

        # DECODER INITIALIZATION
        ###################################

        # Decoder - Block 1
        self._LIFConvBlock(
            128,
            64,
            kernel_size=3,
            alpha=alpha,
            beta=beta,
            alpharp=alpharp,
            deltat=deltat,
            pool_size=2,
            pool_method="up",
            lc_ampl=lc_ampl,
            # 4 channels for readout = (coordinates for bbox)
            readout_channels=4,
            linear_readout_shape=(input_shape[0]/2, input_shape[1]/2)
        )

        # Decoder - Block 2
        self._LIFConvBlock(
            64,
            32,
            kernel_size=3,
            alpha=alpha,
            beta=beta,
            alpharp=alpharp,
            deltat=deltat,
            pool_size=2,
            pool_method="up",
            lc_ampl=lc_ampl,
            readout_channels=4,
            linear_readout_shape=(input_shape[0], input_shape[1])
        )

        # Decoder - Block 3
        self._LIFConvBlock(
            32,
            16,
            kernel_size=3,
            alpha=alpha,
            beta=beta,
            alpharp=alpharp,
            deltat=deltat,
            pool_size=2,
            pool_method="none",
            lc_ampl=lc_ampl,
            readout_channels=4,
            linear_readout_shape=(input_shape[0], input_shape[1])
        )

    def forward(self, input):
        s_out = []
        r_out = []
        u_out = []

        # Encoder forward pass (first three layers of the layer list)
        enc1_lif, enc1_pooling, enc1_readout = (
            self.LIF_layers[0],
            self.poolings[0],
            self.readout_layers[0],
        )
        input = self._forward_LIFConvBlock(
            input, enc1_lif, enc1_pooling, enc1_readout, s_out, r_out, u_out
        )
        enc1_residual = input  # 32 conv with 1/2 resolution

        enc2_lif, enc2_pooling, enc2_readout = (
            self.LIF_layers[1],
            self.poolings[1],
            self.readout_layers[1],
        )
        input = self._forward_LIFConvBlock(
            input, enc2_lif, enc2_pooling, enc2_readout, s_out, r_out, u_out
        )
        enc2_residual = input  # 64 conv with 1/4 resolution

        enc3_lif, enc3_pooling, enc3_readout = (
            self.LIF_layers[2],
            self.poolings[2],
            self.readout_layers[2],
        )
        input = self._forward_LIFConvBlock(
            input, enc3_lif, enc3_pooling, enc3_readout, s_out, r_out, u_out
        )

        # Decoder Forward pass (last three layers of the layer list)
        dec1_lif, dec1_pooling, dec1_readout = (
            self.LIF_layers[3],
            self.poolings[3],
            self.readout_layers[3],
        )
        input = self._forward_LIFConvBlock(
            input, dec1_lif, dec1_pooling, dec1_readout, s_out, r_out, u_out
        )

        # residual connection with enc2 feature maps
        input += nn.functional.interpolate(enc2_residual, scale_factor=2)
        dec2_lif, dec2_pooling, dec2_readout = (
            self.LIF_layers[4],
            self.poolings[4],
            self.readout_layers[4],
        )
        input = self._forward_LIFConvBlock(
            input, dec2_lif, dec2_pooling, dec2_readout, s_out, r_out, u_out
        )

        # residual connection with enc2 feature maps
        input += nn.functional.interpolate(enc1_residual, scale_factor=2)
        dec3_lif, dec3_pooling, dec3_readout = (
            self.LIF_layers[5],
            self.poolings[5],
            self.readout_layers[5],
        )
        input = self._forward_LIFConvBlock(
            input, dec3_lif, dec3_pooling, dec3_readout, s_out, r_out, u_out
        )

        return s_out, r_out, u_out

    def _LIFConvBlock(
        self,
        in_channels,
        out_channels,
        linear_readout_shape,
        kernel_size=3,
        stride=1,
        alpha=0.9,
        beta=0.85,
        alpharp=0.65,
        deltat=1000,
        pool_size=2,
        pool_method="down",
        lc_ampl=0.5,
        readout_channels=2,
    ):
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            # padding_mode="replicate",
            padding=kernel_size // 2,
            stride=stride,
        )

        lif_layer = LIFLayer(
            layer=conv_layer,
            alpha=alpha,
            beta=beta,
            alpharp=alpharp,
            deltat=deltat,
        ).to(device)

        if pool_method == "down":
            pooling = nn.MaxPool2d(kernel_size=pool_size).to(device)
        elif pool_method == "up":
            pooling = nn.Upsample(scale_factor=pool_size).to(device)
        else:
            pooling = None

        readout = nn.Linear(int(
            linear_readout_shape[0] * linear_readout_shape[1] * out_channels), readout_channels).to(device)

        # Readout layer has random fixed weights
        for param in readout.parameters():
            param.requires_grad = False
        self.reset_lc_parameters(readout, lc_ampl)

        # Append the created layers
        self.LIF_layers.append(lif_layer)
        self.poolings.append(pooling)
        self.readout_layers.append(readout)

    def _forward_LIFConvBlock(
        self, input, lif_layer, pooling, readout, s_out, r_out, u_out
    ):
        """
        Performs a forward pass for one LIFConvBlock
        """
        s, u = lif_layer(input)
        u_p = u if pooling is None else pooling(u)
        u_sig = sigmoid(u_p)

        if isinstance(readout, nn.Conv2d):
            r = readout(u_sig)
        else:
            r = readout(u_sig.reshape(u_sig.size(0), -1))

        # non-linearity must be applied after the pooling to ensure that the same winner is selected
        s_out.append(smooth_step(u_p))
        r_out.append(r)
        u_out.append(u_p)
        input = smooth_step(u_p.detach())
        return input


    def _FCLayer(
        self,
        in_channels,
        out_channels,
        linear_readout_shape,
        kernel_size=3,
        stride=1,
        alpha=0.9,
        beta=0.85,
        alpharp=0.65,
        deltat=1000,
        pool_size=2,
        pool_method="down",
        lc_ampl=0.5,
        readout_channels=2,
    ):
        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            # padding_mode="replicate",
            padding=kernel_size // 2,
            stride=stride,
        )

        lif_layer = LIFLayer(
            layer=conv_layer,
            alpha=alpha,
            beta=beta,
            alpharp=alpharp,
            deltat=deltat,
        ).to(device)

        if pool_method == "down":
            pooling = nn.MaxPool2d(kernel_size=pool_size).to(device)
        elif pool_method == "up":
            pooling = nn.Upsample(scale_factor=pool_size).to(device)
        else:
            pooling = None

        readout = nn.Linear(int(
            linear_readout_shape[0] * linear_readout_shape[1] * out_channels), readout_channels).to(device)

        # Readout layer has random fixed weights
        for param in readout.parameters():
            param.requires_grad = False
        self.reset_lc_parameters(readout, lc_ampl)

        # Append the created layers
        self.LIF_layers.append(lif_layer)
        self.poolings.append(pooling)
        self.readout_layers.append(readout)

    def _forward_FCLayer(
        self, input, lif_layer, pooling, readout, s_out, r_out, u_out
    ):
        """
        Performs a forward pass for one LIFConvBlock
        """
        s, u = lif_layer(input)
        u_p = u if pooling is None else pooling(u)
        u_sig = sigmoid(u_p)

        if isinstance(readout, nn.Conv2d):
            r = readout(u_sig)
        else:
            r = readout(u_sig.reshape(u_sig.size(0), -1))

        # non-linearity must be applied after the pooling to ensure that the same winner is selected
        s_out.append(smooth_step(u_p))
        r_out.append(r)
        u_out.append(u_p)
        input = smooth_step(u_p.detach())
        return input