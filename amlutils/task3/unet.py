import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_transforms

import pytorch_lightning as pl


def crop_and_concat(
        conv_out: torch.Tensor, upconv_out: torch.Tensor) -> torch.Tensor:
    '''
    Perform copy, crop and concatenation for skip layers.
    '''
    conv_out_crop = F_transforms.center_crop(conv_out, upconv_out.shape[2:])
    return torch.concat([conv_out_crop, upconv_out], dim=1)


def print_shape(var_name: str, var: torch.Tensor, debug: bool):
    if debug:
        print(f'{var_name}.shape = {var.shape}')


def get_num_incoming_nodes(tensor: torch.Tensor) -> int:
    channels, _, width, height = tensor.shape
    return channels * width * height


def initialize_conv_weights(weights: torch.Tensor):
    std = math.sqrt(2 / get_num_incoming_nodes(weights))
    nn.init.normal_(weights, mean=0.0, std=std)


def initialize_conv_bias(bias: torch.Tensor):
    nn.init.zeros_(bias)


def initialize_conv_relu_layer(layer: nn.Module):
    for name, param in layer.named_parameters():
        if 'weight' in name:
            initialize_conv_weights(param.data)
        if 'bias' in name:
            initialize_conv_bias(param.data)


class UNet(pl.LightningModule):
    '''
    Architecture based on research paper:

    https://arxiv.org/pdf/1505.04597.pdf
    '''

    def __init__(self, learning_rate=1e-3, n_classes=2, debug=False):
        '''
        Initialize layers.

        Args:
            n_classes (int): Number of classes to map to (default=2).
            debug (bool): Print debug information like layer output shapes.
                Defaults to False.
        '''
        super(UNet, self).__init__()

        self.learning_rate = learning_rate
        self.debug = debug

        ## Contracting path. ##
        self.contract_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.contract_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.contract_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.contract_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.contract_conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.contract_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.contract_conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.contract_conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.contract_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.contract_conv7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.contract_conv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.contract_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.contract_conv9 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.contract_conv10 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding='same'),
            nn.ReLU()
            )

        self.contract_dropout = nn.Dropout(0.1)

        ## Expansive path. ##
        self.expand_upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024, 512, kernel_size=2)
            )
        self.expand_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.expand_conv2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU()
            )

        self.expand_upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=2)
            )
        self.expand_conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.expand_conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU()
            )

        self.expand_upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=2)
            )
        self.expand_conv5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.expand_conv6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU()
            )

        self.expand_upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=2)
            )
        self.expand_conv7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.expand_conv8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU()
            )

        self.expand_final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        self.initialize_parameters()

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if name.endswith(".weight"):
                initialize_conv_weights(param.data)
            if name.endswith(".bias"):
                initialize_conv_bias(param.data)

    def forward(self, X):
        ## Contracting path. ##
        print_shape('X', X, self.debug)
        contract_conv1_out = self.contract_conv1(X)
        print_shape('contract_conv1_out', contract_conv1_out, self.debug)
        contract_conv2_out = self.contract_conv2(contract_conv1_out)
        print_shape('contract_conv2_out', contract_conv2_out, self.debug)
        contract_pool1_out = self.contract_pool1(contract_conv2_out)
        print_shape('contract_pool1_out', contract_pool1_out, self.debug)

        contract_conv3_out = self.contract_conv3(contract_pool1_out)
        print_shape('contract_conv3_out', contract_conv3_out, self.debug)
        contract_conv4_out = self.contract_conv4(contract_conv3_out)
        print_shape('contract_conv4_out', contract_conv4_out, self.debug)
        contract_pool2_out = self.contract_pool2(contract_conv4_out)
        print_shape('contract_pool2_out', contract_pool2_out, self.debug)

        contract_conv5_out = self.contract_conv5(contract_pool2_out)
        print_shape('contract_conv5_out', contract_conv5_out, self.debug)
        contract_conv6_out = self.contract_conv6(contract_conv5_out)
        print_shape('contract_conv6_out', contract_conv6_out, self.debug)
        contract_pool3_out = self.contract_pool3(contract_conv6_out)
        print_shape('contract_pool3_out', contract_pool3_out, self.debug)

        contract_conv7_out = self.contract_conv7(contract_pool3_out)
        print_shape('contract_conv7_out', contract_conv7_out, self.debug)
        contract_conv8_out = self.contract_conv8(contract_conv7_out)
        print_shape('contract_conv8_out', contract_conv8_out, self.debug)
        contract_pool4_out = self.contract_pool4(contract_conv8_out)
        print_shape('contract_pool4_out', contract_pool4_out, self.debug)

        contract_conv9_out = self.contract_conv9(contract_pool4_out)
        print_shape('contract_conv9_out', contract_conv9_out, self.debug)
        contract_conv10_out = self.contract_conv10(contract_conv9_out)
        print_shape('contract_conv10_out', contract_conv10_out, self.debug)

        contract_dropout_out = self.contract_dropout(contract_conv10_out)
        print_shape('contract_dropout_out', contract_dropout_out, self.debug)

        ## Expansive path. ##
        expand_upconv1_out = self.expand_upconv1(contract_dropout_out)
        print_shape('expand_upconv1_out', expand_upconv1_out, self.debug)
        expand_conv1_in = crop_and_concat(
            contract_conv8_out, expand_upconv1_out)
        print_shape('expand_conv1_in', expand_conv1_in, self.debug)
        expand_conv1_out = self.expand_conv1(expand_conv1_in)
        print_shape('expand_conv1_out', expand_conv1_out, self.debug)
        expand_conv2_out = self.expand_conv2(expand_conv1_out)
        print_shape('expand_conv2_out', expand_conv2_out, self.debug)

        expand_upconv2_out = self.expand_upconv2(expand_conv2_out)
        print_shape('expand_upconv2_out', expand_upconv2_out, self.debug)
        expand_conv3_in = crop_and_concat(
            contract_conv6_out, expand_upconv2_out)
        print_shape('expand_conv3_in', expand_conv3_in, self.debug)
        expand_conv3_out = self.expand_conv3(expand_conv3_in)
        print_shape('expand_conv3_out', expand_conv3_out, self.debug)
        expand_conv4_out = self.expand_conv4(expand_conv3_out)
        print_shape('expand_conv4_out', expand_conv4_out, self.debug)

        expand_upconv3_out = self.expand_upconv3(expand_conv4_out)
        print_shape('expand_upconv3_out', expand_upconv3_out, self.debug)
        expand_conv5_in = crop_and_concat(
            contract_conv4_out, expand_upconv3_out)
        print_shape('expand_conv5_in', expand_conv5_in, self.debug)
        expand_conv5_out = self.expand_conv5(expand_conv5_in)
        print_shape('expand_conv5_out', expand_conv5_out, self.debug)
        expand_conv6_out = self.expand_conv6(expand_conv5_out)
        print_shape('expand_conv6_out', expand_conv6_out, self.debug)

        expand_upconv4_out = self.expand_upconv4(expand_conv6_out)
        print_shape('expand_upconv4_out', expand_upconv4_out, self.debug)
        expand_conv7_in = crop_and_concat(
            contract_conv2_out, expand_upconv4_out)
        print_shape('expand_conv7_in', expand_conv7_in, self.debug)
        expand_conv7_out = self.expand_conv7(expand_conv7_in)
        print_shape('expand_conv7_out', expand_conv7_out, self.debug)
        expand_conv8_out = self.expand_conv8(expand_conv7_out)
        print_shape('expand_conv8_out', expand_conv8_out, self.debug)

        final_out = self.expand_final_conv(expand_conv8_out)
        print_shape('final_out', final_out, self.debug)

        return final_out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_nb):
        X, y = batch
        X = torch.unsqueeze(X, 1)

        y_hat = self.forward(X)
        y = F_transforms.center_crop(y, y_hat.shape[2:])

        '''
        prediction = F.softmax(y_hat, dim=1)
        prediction = torch.squeeze(prediction)

        X_crop = F_transforms.center_crop(X, y_hat.shape[2:])
        X_crop = torch.squeeze(X_crop)
        X_crop = torch.squeeze(X_crop).numpy()

        visualize_segmentation(
            X_crop,
            prediction[0,:,:].detach().numpy(),
            segmentation_opacity=1
        )

        visualize_segmentation(
            X_crop,
            prediction[1,:,:].detach().numpy(),
            segmentation_opacity=1
        )
        '''

        loss = F.cross_entropy(y_hat, y)

        self.log('train_loss', loss)
        return {'loss': loss}

    # def validation_step(self, batch, batch_nb):
    #    x, y = batch
    #    y_hat = self.forward(x)
    #    loss = F.cross_entropy(y_hat, y)
    #    return {'val_loss': loss}
