from os import environ

if environ['TF']:
    from tensorflow.keras import backend as K


    # Originally from https://github.com/fizyr/keras-retinanet/blob/4d461b4/keras_retinanet/losses.py#L70-L117
    def smooth_l1(sigma=3.0):
        """ Create a smooth L1 loss functor.
        Args
            sigma: This argument defines the point where the loss changes from L2 to L1.
        Returns
            A functor for computing the smooth L1 loss given target data and predicted data.
        """
        sigma_squared = sigma ** 2

        def _smooth_l1(y_true, y_pred):
            """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
            Args
                y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
                y_pred: Tensor from the network of shape (B, N, 4).
            Returns
                The smooth L1 loss of y_pred w.r.t. y_true.
            """
            # separate target and state
            regression = y_pred
            regression_target = y_true[:, :, :-1]
            anchor_state = y_true[:, :, -1]

            # filter out "ignore" anchors
            indices = K.where(K.equal(anchor_state, 1))
            regression = K.gather_nd(regression, indices)
            regression_target = K.gather_nd(regression_target, indices)

            # compute smooth L1 loss
            # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
            #        |x| - 0.5 / sigma / sigma    otherwise
            regression_diff = regression - regression_target
            regression_diff = K.abs(regression_diff)
            regression_loss = K.where(
                K.less(regression_diff, 1.0 / sigma_squared),
                0.5 * sigma_squared * K.pow(regression_diff, 2),
                regression_diff - 0.5 / sigma_squared
            )

            # compute the normalizer: the number of positive anchors
            normalizer = K.maximum(1, K.shape(indices)[0])
            normalizer = K.cast(normalizer, dtype=K.floatx())
            return K.sum(regression_loss) / normalizer

        return _smooth_l1


    SmoothL1 = smooth_l1()
elif environ['TORCH']:
    def SmoothL1(*args, **kwargs):
        raise NotImplementedError()
else:
    def SmoothL1(*args, **kwargs):
        raise NotImplementedError()
