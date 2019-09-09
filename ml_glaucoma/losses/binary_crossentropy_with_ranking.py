# From: https://gist.github.com/jerheff/8cf06fe1df0695806456
def BinaryCrossentropyWithRanking(y_true, y_pred):
    """ Trying to combine ranking loss with numeric precision"""
    import tensorflow.keras.backend as K

    # first get the log loss like normal
    logloss = K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

    # next, build a rank loss

    # clip the probabilities to keep stability
    y_pred_clipped = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    # translate into the raw scores before the logit
    y_pred_score = K.log(y_pred_clipped / (1 - y_pred_clipped))

    # determine what the maximum score for a zero outcome is
    y_pred_score_zerooutcome_max = K.max(y_pred_score * (y_true < 1))

    # determine how much each score is above or below it
    rankloss = y_pred_score - y_pred_score_zerooutcome_max

    # only keep losses for positive outcomes
    rankloss = rankloss * y_true

    # only keep losses where the score is below the max
    rankloss = K.square(K.clip(rankloss, -100, 0))

    # average the loss for just the positive outcomes
    rankloss = K.sum(rankloss, axis=-1) / (K.sum(y_true > 0) + 1)

    # return (rankloss + 1) * logloss - an alternative to try
    return rankloss + logloss
