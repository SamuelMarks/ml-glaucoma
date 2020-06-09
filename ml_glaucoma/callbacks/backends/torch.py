def get_callbacks(
    batch_size,
    checkpoint_freq='epoch',
    summary_freq=10,
    model_dir=None,
    train_steps_per_epoch=None,
    val_steps_per_epoch=None,
    lr_schedule=None,
    tensorboard_log_dir=None,
    write_images=False,
):
    raise NotImplementedError()


__all__ = ['get_callbacks']
