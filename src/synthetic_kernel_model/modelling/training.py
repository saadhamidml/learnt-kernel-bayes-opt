from pathlib import Path
import torch
import gpytorch

from . import models


def train(force_train,
          kernel,
          likelihood_type,
          optimiser_type,
          learning_rate,
          training_iterations,
          train_x,
          train_y,
          log_dir=Path('./')):
    if likelihood_type == 'gaussian':
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if kernel == 'rbf':
        model = models.ExactGPModel(train_x, train_y, likelihood)
    elif kernel == 'spectral_mixture':
        model = models.SpectralMixtureGPModel(train_x, train_y, likelihood)
    elif kernel == 'sparse_spectrum':
        model = models.SparseSpectrumGPModel(train_x, train_y, likelihood)
    if optimiser_type == 'adam':
        optimiser = torch.optim.Adam([{'params': model.parameters()}, ],
                                     lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    if not force_train and (log_dir / 'checkpoint.tar').exists():
        checkpoint = torch.load(log_dir / 'checkpoint.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iter']
    else:
        start_iter = 0
    if start_iter < training_iterations:
        model.train()
        likelihood.train()
        for i in range(start_iter, training_iterations):
            optimiser.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print(f'Iteration: {i + 1}/{training_iterations}',
                  f'    Loss: {loss.item()}')
            optimiser.step()
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'iter': i + 1,
                    'loss': loss},
                   log_dir / 'checkpoint.tar')
    return model, likelihood
