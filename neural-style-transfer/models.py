import torch
import torch.nn as nn
import torchvision.models as models

class VGG16(torch.nn.Module):
    def __init__(self, content_layer = 'relu4_3', 
                style_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=True).requires_grad_(False).features
        
        self.layers = {}
        block = 1
        conv_index = 1
        relu_index = 1
        for index, layer in enumerate(self.vgg16):
            name = layer.__class__.__name__
            if name == 'Conv2d':
                self.layers[f'conv{block}_{conv_index}'] = index
                conv_index += 1
            elif name == 'ReLU':
                self.layers[f'relu{block}_{relu_index}'] = index
                relu_index += 1
            elif name == 'MaxPool2d':
                max_pool = self.vgg16[index]
                self.vgg16[index] = nn.AvgPool2d(kernel_size=max_pool.kernel_size, stride=max_pool.stride,
                                        padding=max_pool.padding, ceil_mode=max_pool.ceil_mode)
                self.layers[f'pool{block}'] = index
                block += 1
                conv_index = 1
                relu_index = 1
            else:
                print('Something is wrong!!!')

        self.content_layer = content_layer
        self.style_layers = style_layers

        for p in self.parameters():
            p.requires_grad = False


    def forward(self, x):
        content_feature_map = None
        style_feature_maps = []

        style_index = 0
        for name, index in self.layers.items():
            x = self.vgg16[index](x)
            if name == self.content_layer:
                content_feature_map = x
            
            if style_index < len(self.style_layers) and name == self.style_layers[style_index]:
                style_index += 1
                style_feature_maps.append(x)

        
        return (content_feature_map, style_feature_maps)

