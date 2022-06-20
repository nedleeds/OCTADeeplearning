SEED = 7

def seed_everything(seed):
    import torch, random, os, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

from copy import deepcopy
from torch import nn
from .modules import (
    CropMaxUnpool2d,
    CropMaxUnpool3d,
    PadMaxPool2d,
    PadMaxPool3d,
    Reshape,
)


class CNN_Transformer(nn.Module):
    def __init__(self, model=None):
        """
        Construct an autoencoder from a given CNN. The encoder part corresponds to the convolutional part of the CNN.

        :param model: (Module) a CNN. The convolutional part must be comprised in a 'features' class variable.
        """
        from copy import deepcopy

        super(CNN_Transformer, self).__init__()

        self.level = 0

        if model is not None:
            self.encoder = deepcopy(model.convolutions)
            self.decoder = self.construct_inv_layers(model)

            for i, sequential in enumerate(self.encoder):
                for layer in sequential:
                    if isinstance(layer, PadMaxPool3d) or isinstance(layer, PadMaxPool2d):
                        self.encoder[i].set_new_return()
                    elif isinstance(layer, nn.MaxPool3d) or isinstance( layer, nn.MaxPool2d):
                        layer.return_indices = True
                                
        else:
            self.encoder = nn.Sequential()
            self.decoder = nn.Sequential()

    def __len__(self):
        return len(self.encoder)

    def construct_inv_layers(self, model):
        """
        Implements the decoder part from the CNN. The decoder part is the symmetrical list of the encoder
        in which some layers are replaced by their transpose counterpart.
        ConvTranspose and ReLU layers are inverted in the end.

        :param model: (Module) a CNN. The convolutional part must be comprised in a 'features' class variable.
        :return: (Module) decoder part of the Autoencoder
        """
        inv_sequential = nn.Sequential()
        len_sequential = len(self.encoder)
        inv_layers = []
        for i, sequential in enumerate(self.encoder):
            inv_layers_list = []
            for layer in sequential:
                if isinstance(layer, nn.Conv3d):
                    inv_layers_list.append(
                        nn.ConvTranspose3d( 
                            layer.out_channels,
                            layer.in_channels,
                            layer.kernel_size,
                            stride=layer.stride,
                            padding=layer.padding,
                        )
                    )
                    self.level += 1
                elif isinstance(layer, nn.Conv2d):
                    inv_layers_list.append(
                        nn.ConvTranspose2d(
                            layer.out_channels,
                            layer.in_channels,
                            layer.kernel_size,
                            stride=layer.stride,
                            padding=layer.padding,
                        )
                    )
                    self.level += 1
                elif isinstance(layer, nn.MaxPool3d):
                    k = layer.kernel_size
                    s = layer.stride
                    p = layer.padding
                    d = layer.dilation
                    inv_layers_list.append(
                        nn.MaxUnpool3d(kernel_size=k, stride=s, padding=p)
                    )
                elif isinstance(layer, PadMaxPool3d):
                    inv_layers_list.append(
                        CropMaxUnpool3d(layer.kernel_size, stride=layer.stride)
                    )
                elif isinstance(layer, PadMaxPool2d):
                    inv_layers_list.append(
                        CropMaxUnpool2d(layer.kernel_size, stride=layer.stride)
                    )
                elif isinstance(layer, nn.Linear):
                    inv_layers_list.append(nn.Linear(layer.out_features, layer.in_features))
                elif isinstance(layer, nn.Flatten):
                    inv_layers_list.append(Reshape(model.flattened_shape))
                elif isinstance(layer, nn.LeakyReLU):
                    inv_layers_list.append(nn.LeakyReLU(negative_slope=1 / layer.negative_slope))
                else:
                    inv_layers_list.append(deepcopy(layer))

            inv_layers_list = self.replace_relu(inv_layers_list)
            inv_layers_list.reverse()
            inv_layers.append(inv_layers_list)

        inv_layers.reverse()
        for i, layers in enumerate(inv_layers):
            inv_layers_seq = nn.Sequential(*layers)
            inv_sequential.add_module(f'convT{len_sequential-i}', inv_layers_seq)
        # return nn.Sequential(*inv_layers)
        return inv_sequential

    @staticmethod
    def replace_relu(inv_layers):
        """
        Invert convolutional and ReLU layers (give empirical better results)

        :param inv_layers: (list) list of the layers of decoder part of the Auto-Encoder
        :return: (list) the layers with the inversion
        """
        idx_relu, idx_conv = -1, -1
        for idx, layer in enumerate(inv_layers):
            if isinstance(layer, nn.ConvTranspose3d):
                idx_conv = idx
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.LeakyReLU):
                idx_relu = idx

            if idx_conv != -1 and idx_relu != -1:
                inv_layers[idx_relu], inv_layers[idx_conv] = (
                    inv_layers[idx_conv],
                    inv_layers[idx_relu],
                )
                idx_conv, idx_relu = -1, -1

        # Check if number of features of batch normalization layers is still correct
        for idx, layer in enumerate(inv_layers):
            if isinstance(layer, nn.BatchNorm3d):
                conv = inv_layers[idx + 1]
                inv_layers[idx] = nn.BatchNorm3d(conv.out_channels)

        return inv_layers