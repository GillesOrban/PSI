# resnet model, modified input and output layers

import torch
import torch.nn as nn
from torchvision import models
import psi.deep_wfs.utils.training_tools as utils
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class Net(nn.Module):

    def __init__(self, n_channels_in, n_channels_out, resnet_archi, weights_path=False):
        super(Net, self).__init__()

        # Use pre-trained weights from ImageNet (overwritten if load state dict later):
        if resnet_archi == "resnet50":
            self.resnet = resnet50(weights = ResNet50_Weights.DEFAULT)
            out_dim = 2048
        elif resnet_archi == "resnet34":
            self.resnet = resnet34(weights = ResNet34_Weights.DEFAULT)
            out_dim = 512
        elif resnet_archi == "resnet18":
            self.resnet = resnet18(weights = ResNet18_Weights.DEFAULT)
            out_dim = 512
        else:
            raise ValueError("The ResNet architecture specified is not valid (must be 'resnet18', 'resnet34' or 'resnet50')")

        # *** Modify first layer (to account for the input dimension):
        # first_conv_layer = [nn.Conv2d(n_channels_in, 3, kernel_size=1, stride=1, bias=True),
        #                     nn.AdaptiveMaxPool2d(224),  # 224x224 -> ImageNet gridsize
        #                     self.resnet.conv1]
        # self.resnet.conv1 = nn.Sequential(*first_conv_layer)

        layer = self.resnet.conv1
                
        # Creating new Conv2d layer
        new_layer = nn.Conv2d(in_channels=n_channels_in, 
                        out_channels=layer.out_channels, 
                        kernel_size=layer.kernel_size, 
                        stride=layer.stride, 
                        padding=layer.padding,
                        bias=layer.bias)

        # Copying the weights from the old to the new layer
        new_layer.weight[:, :n_channels_in, :, :].data[...] = torch.autograd.Variable(layer.weight[:, :n_channels_in, : :].clone(),
                                                                                      requires_grad=True)
        new_layer.weight = nn.Parameter(new_layer.weight)
        self.resnet.conv1 = new_layer

        # *** Modify last layer (to account for the output dimension):
        # self.resnet.fc = nn.Linear(out_dim, n_channels_out)
        # Modify last layer (to account for the output dimension):
        new_last_layer = [nn.Linear(out_dim, 250),
                          nn.ReLU(),
                          nn.Linear(250, 100),
                          nn.ReLU()]
        self.resnet.fc = nn.Sequential(*new_last_layer)

        # Linear layer to map to modes
        self.post_res = nn.Linear(100, n_channels_out)

        # Load weights from previous training (if requested):
        if weights_path is not False:
            # Load specified pre-trained weights:
            pretrained_weights = torch.load(weights_path)
            try:
                state_dict = utils.set_state_dict(pretrained_weights, remove_str_size=7)  # remove 'resnet.'
                self.resnet.load_state_dict(state_dict)
            except RuntimeError:
                # Necessary for loading the weights if trained with several GPUs:
                state_dict = utils.set_state_dict(pretrained_weights, remove_str_size=14)  # remove also 'module.' of dataparallel
                self.resnet.load_state_dict(state_dict)

        # Allows to fine-tune all ResNet weights:
        for param in self.resnet.parameters():
            param.requires_grad = True

    def forward(self, x):
        z = self.resnet(x)
        z = self.post_res(z)
        return z
