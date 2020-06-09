from inspect import currentframe

import gin


@gin.configurable(blacklist=['inputs', 'output_spec'])
def dr0(inputs, output_spec,
        weights='imagenet', pooling='avg', final_activation='default',
        kwargs=None):
    name = currentframe().f_code.co_name
    raise NotImplementedError(name)


del currentframe, gin
