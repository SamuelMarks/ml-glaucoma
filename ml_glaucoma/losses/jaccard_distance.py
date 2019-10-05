from os import environ

if environ['TF']:
    def JaccardDistance(y_true, y_pred, smooth=100):
        """
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

        The jaccard distance loss is useful for unbalanced datasets. This has been
        shifted so it converges on 0 and is smoothed to avoid exploding or disappearing
        gradient.

        Ref: https://en.wikipedia.org/wiki/Jaccard_index

        @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
        @author: wassname
        """
        import tensorflow.keras.backend as K

        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth
elif environ['TORCH']:
    def JaccardDistance(*args, **kwargs):
        raise NotImplementedError()
else:
    def JaccardDistance(*args, **kwargs):
        raise NotImplementedError()
