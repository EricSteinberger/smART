# Copyright(c) Eric Steinberger 2018

import torch
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torchvision import transforms
from torchvision.transforms import functional as FT

from src.style_transfer._VGG import VGG


class NeuralStyleTransfer:

    def __init__(self,
                 content_img_path,
                 style_img_path,
                 vgg_weights_path,
                 output_img_path,

                 content_layers=['4_2'],
                 content_weights=[1.0],
                 style_layers=['1_1', '2_1', '3_1', '4_1', '5_1'],
                 style_weights=[0.25, 0.06, 0.015, 0.004, 0.003],

                 n_iterations=750,
                 optimization_img_size=400,

                 use_gpu=True,
                 ):

        self._content_layers = content_layers
        self._style_layers = style_layers
        self._style_weights = style_weights
        self._content_weights = content_weights
        self._n_iterations = n_iterations
        self._optimization_img_size = optimization_img_size
        self.device = torch.device("cuda:0" if use_gpu else "cpu")

        self._output_img_path = output_img_path

        # Setup pre-trained Deep CNN
        self._cnn = VGG(style_layers=style_layers, content_layers=content_layers)
        self._cnn.load_state_dict(torch.load(vgg_weights_path))
        self._cnn.eval()
        for param in self._cnn.parameters():
            param.requires_grad = False

        # """"""""""""""""""""""""""""""""""""
        # Image processing operations
        # """"""""""""""""""""""""""""""""""""
        self._pre_proc_op = transforms.Compose(
            [
                transforms.Resize(self._optimization_img_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # RGB -> BGR
                transforms.Normalize(mean=[0.408, 0.458, 0.485], std=[1, 1, 1]),
                transforms.Lambda(lambda x: x.mul_(255)),
            ]
        )
        self._post_proc_op = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.mul_(1. / 255)),
                transforms.Normalize(mean=[-0.408, -0.458, -0.4850], std=[1, 1, 1]),
                transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # BGR -> RGB
            ]
        )

        # """"""""""""""""""""""""""""""""""""
        # Import Images
        # """"""""""""""""""""""""""""""""""""
        # Import as PIL images
        self._content_img = Image.open(content_img_path)
        self._style_img = Image.open(style_img_path)

        # Show them
        self._content_img.show()
        self._style_img.show()

        # Preprocess to PyTorch Tensors
        self._content_img = self._pre_proc_op(self._content_img).unsqueeze(0)
        self._style_img = self._pre_proc_op(self._style_img).unsqueeze(0)

        # Create Painting (the image that will be optimized)
        self._painting_img = self._content_img.clone()

        # """"""""""""""""""""""""""""""""""""
        # If you have a GPU, why not use it?
        # """"""""""""""""""""""""""""""""""""
        self._cnn.to(self.device)
        self._content_img = self._content_img.to(self.device)
        self._style_img = self._style_img.to(self.device)
        self._painting_img = self._painting_img.to(self.device)

        # """"""""""""""""""""""""""""""""""""
        # Optimizer and Targets
        # """"""""""""""""""""""""""""""""""""
        self._optimizer = optim.LBFGS([self._painting_img.requires_grad_()])

        # compute optimization targets
        self._style_targets = [_get_gram(feature_layer).detach() for feature_layer in self._cnn(self._style_img)["style"]]
        self._content_targets = [A.detach() for A in self._cnn(self._content_img)["content"]]

    def run(self):
        iter_count = [0]
        while iter_count[0] <= self._n_iterations:

            def closure():
                self._optimizer.zero_grad()
                loss = self._loss()
                loss.backward()
                iter_count[0] += 1

                if iter_count[0] % 20 == 19:
                    print("Iteration ", iter_count[0] + 1, "Current combined loss: ", loss.item())
                return loss

            self._optimizer.step(closure)

        o = self._postprocess_img(self._painting_img.cpu().squeeze())
        o.show()
        with open(self._output_img_path, 'wb') as file:
            o.save(file, 'jpeg')

    def _loss(self):
        layers = self._cnn(self._painting_img)
        style_loss = sum(
            [
                self._style_weights[i] * F.mse_loss(_get_gram(feature_map), self._style_targets[i])
                for i, feature_map in enumerate(layers["style"])
            ]
        )
        content_loss = sum(
            [
                self._content_weights[i] * F.mse_loss(feature_map, self._content_targets[i])
                for i, feature_map in enumerate(layers["content"])
            ]
        )
        return style_loss + content_loss

    def _preprocess_img(self, img):
        return self._pre_proc_op(img)

    def _postprocess_img(self, img):
        x = self._post_proc_op(img=img)
        x = torch.clamp(x, 0, 1)  # NST might kick values into illegal areas
        return FT.to_pil_image(x)


def _get_gram(inp):
    b, c, x, y = inp.size()
    features = inp.view(b, c, x * y)  # Flatten last dim
    gram = torch.bmm(features, features.transpose(1, 2))  # Compute Gram
    return gram.div_(x * y)  # Scale down
