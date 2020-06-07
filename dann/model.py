import torch
import torch.nn as nn
from torch.autograd import Function

class ReverseGradLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.alpha
        return grad_input, None

class SVHNConvNet(nn.Module):
    def __init__(self, nc, nclasses, ndomains):
        super(SVHNConvNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=64, kernel_size=(5, 5), stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=1, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.class_predictor = nn.Sequential(
            nn.Linear(128 * 5 * 5, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(3072, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(2048, nclasses)
        )

        self.domain_predictor = nn.Sequential(
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(1024, ndomains)
        )

    def forward(self, x, alpha):
        feat = self.feature_extractor(x)
        feat = feat.view(x.shape[0], -1)
        reverse_feat = ReverseGradLayer.apply(feat, alpha)
        out_class = self.class_predictor(feat)
        out_domain = self.domain_predictor(reverse_feat)

        return out_class, out_domain
