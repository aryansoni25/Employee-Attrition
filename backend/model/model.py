import torch
import torch.nn as nn


class GaussianMLEstimatorNN(nn.Module):

    def __init__(self, num_features, num_classes):
        super(GaussianMLEstimatorNN, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.mu = nn.Parameter(
            torch.randn(num_classes, num_features)
        )
        self.log_sigma = nn.Parameter(
            torch.zeros(num_classes, num_features)
        )
        self.log_prior = nn.Parameter(
            torch.zeros(num_classes)
        )


    def forward(self, x):
        sigma = torch.exp(self.log_sigma)
        x = x.unsqueeze(1)
        mu = self.mu.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
        log_likelihood = -0.5 * (
            torch.log(2 * torch.pi * sigma**2) +
            ((x - mu) ** 2) / (sigma ** 2)
        )
        log_likelihood = log_likelihood.sum(dim=2)
        log_posterior = log_likelihood + self.log_prior

        return log_posterior