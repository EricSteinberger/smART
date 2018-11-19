# Copyright(c) Eric Steinberger 2018

from torch import nn


class VGG(nn.Module):
    """
    Wraps the pre-trained Deep CNN and allows the extraction of all feature layer activation maps
    """

    def __init__(self, style_layers=None, content_layers=None):
        super().__init__()

        self._content_layers = content_layers
        self._style_layers = style_layers

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self._pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self._relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = {}

        y['1_1'] = self._relu(self.conv1_1(x))
        y['1_2'] = self._relu(self.conv1_2(y['1_1']))

        x = self._pool(y['1_2'])
        y['2_1'] = self._relu(self.conv2_1(x))
        y['2_2'] = self._relu(self.conv2_2(y['2_1']))

        x = self._pool(y['2_2'])
        y['3_1'] = self._relu(self.conv3_1(x))
        y['3_2'] = self._relu(self.conv3_2(y['3_1']))
        y['3_3'] = self._relu(self.conv3_3(y['3_2']))
        y['3_4'] = self._relu(self.conv3_4(y['3_3']))

        x = self._pool(y['3_4'])
        y['4_1'] = self._relu(self.conv4_1(x))
        y['4_2'] = self._relu(self.conv4_2(y['4_1']))
        y['4_3'] = self._relu(self.conv4_3(y['4_2']))
        y['4_4'] = self._relu(self.conv4_4(y['4_3']))

        x = self._pool(y['4_4'])
        y['5_1'] = self._relu(self.conv5_1(x))
        y['5_2'] = self._relu(self.conv5_2(y['5_1']))
        y['5_3'] = self._relu(self.conv5_3(y['5_2']))
        y['5_4'] = self._relu(self.conv5_4(y['5_3']))

        # don't need to apply pool5.

        return {
            "content": [y[key] for key in self._content_layers],
            "style": [y[key] for key in self._style_layers]
        }
