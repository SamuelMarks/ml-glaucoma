import numpy as np
from sklearn.metrics import confusion_matrix


def output_sensitivity_specificity(epoch, predictions, y_test, class_mode='binary'):
    if class_mode == 'binary':
        # determine positive class predictions
        idx = predictions >= 0.5
        predictions = np.zeros(predictions.shape)
        predictions[idx] = 1
        # no need to modify y_test since it consists of zeros and ones already
    else:
        y_test = np.argmax(y_test, axis=-1)
        predictions = np.argmax(predictions, axis=-1)

    c = confusion_matrix(y_test, predictions)
    print('Confusion matrix:\n', c)
    print('[{:03d}] sensitivity'.format(epoch), c[0, 0] / (c[0, 1] + c[0, 0]))
    print('[{:03d}] specificity'.format(epoch), c[1, 1] / (c[1, 1] + c[1, 0]))
