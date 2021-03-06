import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.ops import math_ops, state_ops


# From: https://github.com/4rtemi5/Yogi-Optimizer_Keras/blob/6a2d6ed/yogi_opt.py
class Yogi(Optimizer):
    """Yogi optimizer.
    Default parameters follow those provided in the original paper.
    Arguments:
      lr: float >= 0. Learning rate.
      beta_1: float, 0 < beta < 1. Generally close to 1.
      beta_2: float, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.
      amsgrad: boolean. Whether to apply the AMSGrad variant of this
          algorithm from the paper "On the Convergence of Adam and
          Beyond".
    """

    def __init__(
        self,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.00000001,
        amsgrad=False,
        **kwargs
    ):
        super(Yogi, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype="int64", name="iterations")
            self.lr = K.variable(lr, name="lr")
            self.beta_1 = K.variable(beta_1, name="beta_1")
            self.beta_2 = K.variable(beta_2, name="beta_2")
            self.decay = K.variable(decay, name="decay")
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (  # pylint: disable=g-no-augmented-assignment
                1.0
                / (
                    1.0
                    + self.decay * math_ops.cast(self.iterations, K.dtype(self.decay))
                )
            )

        t = math_ops.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (
            K.sqrt(1.0 - math_ops.pow(self.beta_2, t))
            / (1.0 - math_ops.pow(self.beta_1, t))
        )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1.0 - self.beta_1) * g
            # v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g) # from amsgrad
            v_t = v - (1 - self.beta_2) * K.sign(
                v - math_ops.square(g)
            ) * math_ops.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, "constraint", None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            "lr": float(K.get_value(self.lr)),
            "beta_1": float(K.get_value(self.beta_1)),
            "beta_2": float(K.get_value(self.beta_2)),
            "decay": float(K.get_value(self.decay)),
            "epsilon": self.epsilon,
            "amsgrad": self.amsgrad,
        }
        base_config = super(Yogi, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


__all__ = ["Yogi"]
