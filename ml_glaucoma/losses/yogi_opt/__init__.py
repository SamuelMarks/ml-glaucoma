from os import environ

if environ['TF']:
    from ml_glaucoma.losses.yogi_opt.tf_keras import Yogi
elif environ['TORCH']:
    from ml_glaucoma.losses.yogi_opt.torch import Yogi
else:
    from ml_glaucoma.losses.yogi_opt.other import Yogi
