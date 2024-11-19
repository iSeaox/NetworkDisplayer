import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

import numpy as np

import random

def _hex_to_rgb(hex):
    """! Static and private function that convert hex color value into RBG array

    @param hex  (int)               Hex color value
    @return     (numpy.ndarray)     Array of RBG color's components
    """
    hex = hex.split("#")[-1]
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i:i+2], 16)
        rgb.append(decimal)

    return np.array(rgb)

def _get_color_interval(value, base_hex, offset_hex):
    """! Static and private function that compute color based on print value

    @param value        (float)     Value used to interpol between colors
    @param base_hex     (int)       Base color
    @param offset_hex   (int)       Offset color
    @return             (str)       Interpolated hex color format ("#RRGGBB")
    """
    if value < 0.0:
        value = 0.0
    elif value > 1:
        value = 1.0

    base_color = _hex_to_rgb(base_hex)
    offset_color = _hex_to_rgb(offset_hex)

    deltaColor = offset_color - base_color

    return '#{:02x}{:02x}{:02x}'.format(int(base_color[0] + deltaColor[0] * value), int(base_color[1] + deltaColor[1] * value), int(base_color[2] + deltaColor[2] * value))



class NetworkDisplayer:
    """@brief Provides method to display neural network using matplotlib
    @TODO Make it work with more than 2 classes
    """
    def __init__(self, netClass=(0, 1)):
        self.netClass = netClass
        self.classSpan = abs(netClass[1] - netClass[0])


    def _normalize_point(self, value):
        """! @brief Normalize net output value in range [0, 1] before color interpolation

        @param value (float) Network output
        """
        return abs(self.netClass[1] - value) / self.classSpan


    def draw_point(self, xi, yi, color1, color2):
        """! @brief Plot point after performing color interpolation

        @param xi       (list like) Represent entry 2D vector of neural network
        @param yi       (list like) Represent output 1D vector of neural network
        @param color1   (int)       Base color used for output color interpolation
        @param color2   (int)       Offset color used for output color interpolation
        """
        gen_color = _get_color_interval(self._normalize_point(yi[0]), color1, color2)
        plt.plot([xi[0][0]], [xi[0][1]], "o", color=gen_color)


    def draw_net_repr(self, net, color1, color2, nb_points=10000, bounds=(-1, -1, 2, 2)):
        """! @brief Plot representation of a neural network outputs within bounds

        @param net          (torch.nn.Module extended class)    Neural Network object
        @param color1       (int)                               Base color used for output color interpolation
        @param color2       (int)                               Offset color used for output color interpolation
        @param nb_points    (int)[optional]                     Number of random rendered point. The default value is 10 0000
        @param bounds       (tuple)[optional]                   Viewport like (min_x, min_y, max_x, max_y)
        """
        min_xi_x, min_xi_y, max_xi_x, max_xi_y = bounds
        width = abs(min_xi_x - max_xi_x)
        height = abs(min_xi_y - max_xi_y)

        for i in range(0, nb_points):
            xi = Variable(torch.Tensor([[random.random() * width + min_xi_x, random.random() * height + min_xi_y]]))
            yi = net(xi)

            self.draw_point(xi, yi, color1, color2)


    def draw_inputs(self, inputs, targets, color1, color2):
        """! @brief Plot training data (termed inputs, targets)

        @param inputs   (list<torch.Tensor>)    Input list of training data
        @param targets  (list<torch.Tensor>)    Output list of training data
        @param color1   (int)                   Base color used for output color interpolation
        @param color2   (int)                   Offset color used for output color interpolation
        """
        for input, target in zip(inputs, targets):
            if(target[0] == self.netClass[0]):
                plt.plot([input[0][0]], [input[0][1]], "o", color= color1)
            else:
                plt.plot([input[0][0]], [input[0][1]], "o", color= color2)

    def plot_show(self):
        """! @brief Show the plot
        """
        plt.show()


