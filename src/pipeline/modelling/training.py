from pathlib import Path
import torch
import gpytorch

from . import models


def train(
        model,
        optimiser,
        mll,
        training_iterations,
        force_train,
        train_x,
        train_y,
        log_dir=Path('./'),
        **kwargs
):
    if not force_train and (log_dir / 'checkpoint.tar').exists():
        checkpoint = torch.load(log_dir / 'checkpoint.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['iter']
    else:
        start_iter = 0
    if start_iter < training_iterations:
        model.train()
        # likelihood.train()
        for i in range(start_iter, training_iterations):
            optimiser.zero_grad()
            output = model(train_x)
            loss = -mll(output, model.train_targets)
            loss.backward()
            print(f'Iteration: {i + 1}/{training_iterations}',
                  f'    Loss: {loss.item()}')
            optimiser.step()
        # torch.save({'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimiser.state_dict(),
        #             'iter': i + 1,
        #             'loss': loss},
        #            log_dir / 'checkpoint.tar')
    # return model, likelihood
