from inspect import currentframe

import gin


@gin.configurable(blacklist=['inputs', 'output_spec'])
def se_resnet(inputs, output_spec, application='SEResNet50',
              weights='imagenet', pooling='avg', final_activation='default',
              kwargs=None):
    name = '_'.join((currentframe().f_code.co_name, application))
    raise NotImplementedError(name)


del currentframe
del gin
