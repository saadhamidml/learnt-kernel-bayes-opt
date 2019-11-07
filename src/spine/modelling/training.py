def train(force_train,
          model,
          loss_function,
          optimiser,
          epochs,
          beta,
          train_dl,
          test_dl,
          log_dir=Path('./'),
          **kwargs):
    # Train the model
    if not force_train and (restore or (log_dir / 'checkpoint.tar').exists()):
        checkpoint = torch.load(log_dir / 'checkpoint.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    if start_epoch < epochs:
        fit((start_epoch, epochs),
            model,
            loss_function,
            optimiser,
            train_dl,
            test_dl,
            beta=beta,
            log_dir=log_dir,
            **kwargs)
    return model