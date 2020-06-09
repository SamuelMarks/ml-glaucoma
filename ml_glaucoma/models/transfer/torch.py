from inspect import currentframe

import gin


@gin.configurable(blacklist=['inputs', 'output_spec'])
def transfer_model(inputs, output_spec, transfer='ResNet50',
                   weights='imagenet', pooling='avg', final_activation='default',
                   kwargs=None):
    name = '_'.join((currentframe().f_code.co_name, transfer))
    raise NotImplementedError(name)


del currentframe, gin

__all__ = ['transfer_model']
