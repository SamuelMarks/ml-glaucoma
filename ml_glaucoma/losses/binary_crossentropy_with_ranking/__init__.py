from os import environ

if environ['TF']:
    from ml_glaucoma.losses.binary_crossentropy_with_ranking.tf_keras import BinaryCrossentropyWithRanking
elif environ['TORCH']:
    from ml_glaucoma.losses.binary_crossentropy_with_ranking.torch import BinaryCrossentropyWithRanking
else:
    from ml_glaucoma.losses.binary_crossentropy_with_ranking.other import BinaryCrossentropyWithRanking
