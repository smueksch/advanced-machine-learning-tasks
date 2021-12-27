from comet_ml import Experiment
import math

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_transforms

import pytorch_lightning as pl

from .visualization import visualize_segmentation


def crop_and_concat(
        conv_out: torch.Tensor, upconv_out: torch.Tensor) -> torch.Tensor:
    '''
    Perform copy, crop and concatenation for skip layers.
    '''
    conv_out_crop = F_transforms.center_crop(conv_out, upconv_out.shape[2:])
    return torch.concat([conv_out_crop, upconv_out], dim=1)
    # return torch.concat([conv_out, upconv_out], dim=1)


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

    def __init__(self, experiment: Experiment = None,
                 pixel_weights: torch.Tensor = None, learning_rate=1e-3,
                 n_classes=2, debug=False):
        '''
        Initialize layers.

        Args:
            experiment (Experiment): Comet.ml experiment for logging.
            pixel_weights (torch.Tensor): If supplied, will act as per-pixel
                weights for the cross-entropy loss, see Eq. (1) in linked paper.
            learning_rate (float): Learning rate for training.
            n_classes (int): Number of classes to map to (default=2).
            debug (bool): Print debug information like layer output shapes.
                Defaults to False.
        '''
        super(UNet, self).__init__()

        self.experiment = experiment
        self.pixel_weights = pixel_weights
        self.learning_rate = learning_rate
        self.debug = debug

        # Set up figure for visualization during training.
        self.fig, self.ax = plt.subplots(1, 1, figsize=(6, 6))
        self.fig.set_facecolor('white')
        self.epoch = 0

        ## Contracting path. ##
        self.contract_additional_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.contract_additional_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.contract_additional_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.contract_conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
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
            nn.Conv2d(1024, 512, kernel_size=2, padding='same')
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
            nn.Conv2d(512, 256, kernel_size=2, padding='same')
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
            nn.Conv2d(256, 128, kernel_size=2, padding='same')
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
            nn.Conv2d(128, 64, kernel_size=2, padding='same')
            )
        self.expand_conv7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.expand_conv8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU()
            )

        self.expand_additional_upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=2, padding='same')
            )
        self.expand_additional_conv1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding='same'),
            nn.ReLU()
            )
        self.expand_additional_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.ReLU()
            )

        self.expand_final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

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
        contract_additional_conv1_out = self.contract_additional_conv1(X)
        print_shape('contract_additional_conv1_out',
                    contract_additional_conv1_out, self.debug)
        contract_additional_conv2_out = self.contract_additional_conv2(
            contract_additional_conv1_out)
        print_shape('contract_additional_conv2_out',
                    contract_additional_conv2_out, self.debug)
        contract_additional_pool1_out = self.contract_additional_pool1(
            contract_additional_conv2_out)
        print_shape('contract_additional_pool1_out',
                    contract_additional_pool1_out, self.debug)

        contract_conv1_out = self.contract_conv1(contract_additional_pool1_out)
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

        expand_additional_upconv1_out = self.expand_additional_upconv1(
            expand_conv8_out)
        print_shape('expand_additional_upconv1_out',
                    expand_additional_upconv1_out, self.debug)
        expand_additional_conv1_in = crop_and_concat(
            contract_additional_conv2_out, expand_additional_upconv1_out)
        print_shape('expand_additional_conv1_in',
                    expand_additional_conv1_in, self.debug)
        expand_additional_conv1_out = self.expand_additional_conv1(
            expand_additional_conv1_in)
        print_shape('expand_additional_conv1_out',
                    expand_additional_conv1_out, self.debug)
        expand_additional_conv2_out = self.expand_additional_conv2(
            expand_additional_conv1_out)
        print_shape('expand_additional_conv2_out',
                    expand_additional_conv2_out, self.debug)

        final_out = self.expand_final_conv(expand_additional_conv2_out)
        print_shape('final_out', final_out, self.debug)

        return final_out

    def predict_by_threshold(self, X, threshold: float):
        '''
        Output predicted label according to threshold.

        Threshold must be between 0 and 1.
        '''
        y_hat = self.forward(X)
        prediction = F.softmax(y_hat, dim=1)
        prediction = torch.squeeze(prediction)
        prediction = torch.where(
            prediction[1, :, :] > threshold,
            torch.ones(prediction.shape[1:]),
            torch.zeros(prediction.shape[1:])
            )
        return prediction

    def predict_prob(self, X):
        '''
        Output class probabilities.
        '''
        y_hat = self.forward(X)
        prediction = F.softmax(y_hat, dim=1)
        prediction = torch.squeeze(prediction)
        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_nb):
        X, y = batch
        X = torch.unsqueeze(X, 1)

        y_hat = self.forward(X)
        y = F_transforms.center_crop(y, y_hat.shape[2:])

        if self.pixel_weights is None:
            # No per-pixel weights supplied, compute unweighted cross-entropy.
            loss = F.cross_entropy(y_hat, y)
            self.log('train_loss', loss)
        else:
            # Per-pixel weights supplied, compute weighted cross-entropy.
            y_hat = torch.squeeze(y_hat)
            log_probs = F.log_softmax(y_hat, dim=0)

            selected_log_probs = torch.where(
                y == 1,
                log_probs[1, :, :],
                log_probs[0, :, :]
                )
            pixel_weights = F_transforms.center_crop(
                self.pixel_weights, y_hat.shape[1:])

            weighted_cross_entropy = torch.sum(
                pixel_weights * selected_log_probs)

            # Compute regularization term as Frobenius norm of actual label
            # matrix and the class 1 probabilities to encourage network to not
            # predict too many pixels as class 1 that really aren't.
            probs = F.softmax(y_hat, dim=0)
            class_1_probs = probs[1, :, :]

            regularizer = torch.sum((y - class_1_probs) ** 2)

            loss = -(weighted_cross_entropy + regularizer)

            self.log('train_loss', loss)

        return {'loss': loss, 'X': X}

    def training_epoch_end(self, training_step_outputs):
        if self.experiment is not None:
            last_step_out = training_step_outputs[-1]

            self.experiment.log_epoch_end(self.epoch)

            self.experiment.log_metric(
                'train_loss', last_step_out['loss'],
                epoch=self.epoch)

            # Log visualization of current probability density for class 1.
            threshold = 0.5
            self.freeze()
            prediction = self.predict_by_threshold(
                last_step_out['X'], threshold=threshold)
            self.unfreeze()

            X_crop = F_transforms.center_crop(
                last_step_out['X'], prediction.shape)
            X_crop = torch.squeeze(X_crop)
            X_crop = torch.squeeze(X_crop).numpy()

            self.ax.clear()
            visualize_segmentation(
                self.ax,
                X_crop,
                prediction.numpy(),
                )
            self.ax.set_title(f'Prediction (Threshold {threshold})')

            self.fig.tight_layout()
            self.fig.savefig('prediction.png')

            self.experiment.log_image(
                'prediction.png',
                name=f'prediction-epoch-{self.epoch:03}')

        self.epoch += 1

    # def validation_step(self, batch, batch_nb):
    #    x, y = batch
    #    y_hat = self.forward(x)
    #    loss = F.cross_entropy(y_hat, y)
    #    return {'val_loss': loss}
