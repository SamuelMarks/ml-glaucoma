import traceback
from collections import namedtuple
from functools import reduce
from os import path

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ml_glaucoma.cli_options.hyperparameters import SUPPORTED_LOSSES, SUPPORTED_OPTIMIZERS
from ml_glaucoma.datasets.tfds_builders.dr_spoc import dr_spoc_datasets
from ml_glaucoma.models import valid_models
from ml_glaucoma.utils import update_d, lcs

valid_models_upper = frozenset((model.upper() for model in valid_models))

valid_losses = frozenset(SUPPORTED_LOSSES)
valid_losses_upper = frozenset((model.upper() for model in valid_losses))

valid_optimizers = frozenset(SUPPORTED_OPTIMIZERS)
valid_optimizers_upper = frozenset((model.upper() for model in valid_optimizers))

valid_bases = frozenset(('dc0', 'dc1', 'dc2', 'dc3', 'dc4', 'dr0'))
valid_bases_upper = frozenset((model.upper() for model in valid_bases))


def parse_line(line):  # type: (str) -> ParsedLine
    line = path.basename(line)
    optimizer_params = {}
    if line.count(' ') == 0:
        name, epoch, value = line, 0, 0
    else:
        name, epoch, value = filter(None, line.rstrip().split())
        name, epoch, value = name.rstrip(), int(epoch[6:-3]), float(value)

    split_name = name.split('_')
    ds = split_name[0]

    if split_name[-1].startswith('again') or split_name[-1].endswith('AUC'):
        del split_name[-1]

    def _get_name(st, it):
        return next(s for s in it if s.upper() == st.upper())

    ###########
    #         #
    # dataset #
    #         #
    ###########
    if ds == 'dr':
        ds = max((lcs(line, ds) for ds in dr_spoc_datasets), key=len)

    for idx, word in enumerate(split_name, 2):
        prev_prev_word = split_name[idx - 2] if len(split_name) > idx - 2 else ''
        previous_word = split_name[idx - 1] if len(split_name) > idx - 1 else ''
        prev_prev_word_upper = prev_prev_word.upper()
        previous_word_upper = previous_word.upper()

        upper3 = (prev_prev_word + previous_word + word).upper()
        upper2 = (prev_prev_word + previous_word).upper()
        upper1 = (previous_word + word).upper()
        upper0 = word.upper()

        extra_loss = lambda: ''.join(split_name[(split_name.index(transfer)
                                                 if 'transfer' in locals() and transfer in split_name else 0) + 1:-2]).upper()

        ##########
        #        #
        # epochs #
        #        #
        ##########
        if prev_prev_word == 'epochs':
            epochs = int(previous_word)

        ############
        #          #
        # transfer #
        #          #
        ############
        elif upper0 in valid_models_upper:
            transfer = word
        elif upper1 in valid_models_upper:
            transfer = _get_name(upper1, valid_models)
        elif upper2 in valid_models_upper:
            transfer = _get_name(upper2, valid_models)
        elif upper3 in valid_models_upper:
            transfer = _get_name(upper3, valid_models)
        elif previous_word_upper in valid_models_upper:
            transfer = _get_name(previous_word_upper, valid_models)
        elif prev_prev_word_upper in valid_models_upper:
            transfer = _get_name(prev_prev_word_upper, valid_models)

        ########
        #      #
        # loss #
        #      #
        ########
        elif upper0 in valid_losses_upper:
            loss = word
        elif upper1 in valid_losses_upper:
            loss = _get_name(upper1, valid_losses)
        elif upper2 in valid_losses_upper:
            loss = _get_name(upper2, valid_losses)
        elif upper3 in valid_losses_upper:
            loss = _get_name(upper3, valid_losses)
        elif previous_word_upper in valid_losses_upper:
            loss = _get_name(previous_word_upper, valid_losses)
        elif prev_prev_word_upper in valid_losses_upper:
            loss = _get_name(prev_prev_word_upper, valid_losses)
        elif extra_loss() in valid_losses_upper:
            loss = _get_name(extra_loss(), valid_losses)

        #############
        #           #
        # optimizer #
        #           #
        #############
        elif upper0 in valid_optimizers_upper:
            optimizer = _get_name(word, valid_optimizers)
        elif upper1 in valid_optimizers_upper:
            optimizer = _get_name(upper1, valid_optimizers)
        elif upper2 in valid_optimizers_upper:
            optimizer = _get_name(upper2, valid_optimizers)
        elif upper3 in valid_optimizers_upper:
            optimizer = _get_name(upper3, valid_optimizers)
        elif previous_word_upper in valid_optimizers_upper:
            optimizer = _get_name(previous_word_upper, valid_optimizers)
        elif prev_prev_word_upper in valid_optimizers_upper:
            optimizer = _get_name(prev_prev_word_upper, valid_optimizers)


        ####################
        #                  #
        # optimizer params #
        #                  #
        ####################
        elif previous_word in frozenset(('lr', 'alpha')) and len(split_name) != idx:
            optimizer_params[previous_word] = float(split_name[idx])


        ########
        #      #
        # base #
        #      #
        ########
        elif upper0 in valid_bases_upper:
            base = word
        elif upper1 in valid_bases_upper:
            base = _get_name(upper1, valid_bases)
        elif previous_word_upper in valid_bases_upper:
            base = _get_name(previous_word_upper, valid_bases)

        ########
        #      #
        # else #
        #      #
        ########
        '''
        else:
            print('------------------------------')
            print('prev_prev_word:'.ljust(14), prev_prev_word,
                  '\nprevious_word:'.ljust(16), previous_word,
                  '\nword:'.ljust(16), word,
                  '\nsplit_name:'.ljust(16), split_name,
                  '\nidx:'.ljust(16), idx,
                  '\nlen(split_name):'.ljust(16), len(split_name),
                  '\n------------------------------')
        '''
        assert locals().get('loss') != 'loss', line

        # print('split_name:'.ljust(16), split_name)
    '''
    print(
        'best_epoch:'.ljust(14), epoch,
        '\nbest_auc:'.ljust(15), value,
        '\ntotal_epochs:'.ljust(15), locals().get('epochs'),
        '\ntransfer:'.ljust(15), locals().get('transfer'),
        '\nloss:'.ljust(15), locals().get('loss', 'BinaryCrossentropy'),
        '\noptimizer:'.ljust(15), locals().get('optimizer', 'Adam'), '\n',
        sep=''
    )
    '''

    base, transfer = locals().get('base'), locals().get('transfer')
    assert base is not None or transfer is not None, 'Unknown model'

    return ParsedLine(dataset=ds,
                      epoch=epoch,
                      value=value,
                      epochs=locals().get('epochs'),
                      transfer=transfer,
                      loss=locals().get('loss', 'BinaryCrossentropy'),
                      optimizer=locals().get('optimizer', 'Adam'),
                      optimizer_params=update_d({'lr': 1e-3}, optimizer_params),
                      base=base or 'transfer')


ParsedLine = namedtuple('ParsedLine', ('dataset', 'epoch', 'value', 'epochs', 'transfer',
                                       'loss', 'optimizer', 'optimizer_params', 'base'))

options_space = {
    'last_idx': 2,
    'space': [
        ParsedLine(dataset='refuge',
                   epoch=64,
                   value='value',
                   epochs=250,
                   transfer='Resnet50',
                   loss='BinaryCrossentropy',
                   optimizer='Adam',
                   optimizer_params={'lr': 1e-3},
                   base='transfer'),
        ParsedLine(dataset='refuge',
                   epoch=64,
                   value='value',
                   epochs=250,
                   transfer='MobileNet',
                   loss='CategoricalCrossentropy',
                   optimizer='Nestrov',
                   optimizer_params={'lr': 1e-5},
                   base='transfer')
    ]
}


# Extraction function from https://gist.github.com/ptschandl/ef67bbaa93ec67aba2cab0a7af47700b
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame

    Parameters
    ----------
    path : str
        path to tensorflow log file

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 14072,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            runlog_data = pd.concat([runlog_data,
                                     pd.DataFrame({"metric": [tag] * len(step), "value": values, "step": step})])
    # Dirty catch of DataLossError
    except:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


# Function from https://gist.github.com/ptschandl/ef67bbaa93ec67aba2cab0a7af47700b
def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


def get_metrics(logs, prefix='epoch_val_', tag='auc', total_epochs=250,
                metrics=('loss', 'auc', 'tp50', 'fp50', 'tn50', 'fn50', 'f150')):
    def get_metrics_from_one_logfile(log):
        df = many_logs2pandas((log,))
        df.rename_axis('epoch', inplace=True)
        df['epoch'] = df.index
        df.reset_index(drop=True, inplace=True)

        # metric_with_epoch_greater_than_threshold = df[df['epoch'] > total_epochs]['metric'].unique()
        df = df[df['epoch'] < total_epochs + 1]

        max_metrics = df.loc[df.groupby('metric')['value'].idxmax()]

        # epochs = max_metrics['epoch'].unique()

        best_epoch = max_metrics.loc[
            max_metrics['metric'] == '{}{}'.format(prefix, tag)]['epoch'].iloc[0]
        metrics_of_best_epoch = df[df['epoch'] == best_epoch]

        current_metrics = {
            attr: metrics_of_best_epoch[
                metrics_of_best_epoch['metric'] == '{}{}'.format(prefix, attr)]['value'].iloc[0]
            for attr in metrics
        }

        if len(frozenset(('tp50', 'fp50', 'tn50', 'fn50')) - frozenset(metrics)) == 0:
            loss, auc, tp, fp, tn, fn, f1 = (
                current_metrics[attr]
                for attr in ('loss', 'auc', 'tp50', 'fp50', 'tn50', 'fn50', 'f150')
            )

            current_metrics.update({
                'acc': np.divide(np.add(tp, tn), np.sum((tp, tn, fp, fn))),
                'sensitivity': np.divide(tp, np.add(tp, fn)),
                'specificity': np.divide(tn, np.add(tn, fp))
            })

        return log, namedtuple('Metrics', sorted(current_metrics.keys()))(**current_metrics)

    return reduce(lambda p, c: update_d(p, {c[0]: c[1]}), map(get_metrics_from_one_logfile, logs), {})
