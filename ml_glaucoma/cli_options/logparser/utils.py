from ml_glaucoma.cli_options.hyperparameters import SUPPORTED_LOSSES, SUPPORTED_OPTIMIZERS
from ml_glaucoma.models import valid_models
from ml_glaucoma.utils import update_d

valid_models_upper = frozenset((model.upper() for model in valid_models))

valid_losses = frozenset(SUPPORTED_LOSSES)
valid_losses_upper = frozenset((model.upper() for model in valid_losses))

valid_optimizers = frozenset(SUPPORTED_OPTIMIZERS)
valid_optimizers_upper = frozenset((model.upper() for model in valid_optimizers))

valid_bases = frozenset(('dc0', 'dc1', 'dc2', 'dc3', 'dc4', 'dr0'))
valid_bases_upper = frozenset((model.upper() for model in valid_bases))


def parse_line(line):
    optimizer_params = {}
    name, epoch, value = filter(None, line.rstrip().split())
    name, epoch, value = name.rstrip(), int(epoch[6:-3]), float(value)

    split_name = name.split('_')
    ds = split_name[0]

    if split_name[-1].startswith('again') or split_name[-1].endswith('AUC'):
        del split_name[-1]

    def _get_name(st, it):
        return next(s for s in it if s.upper() == st.upper())

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
            epochs = previous_word

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

    assert locals().get('base') is not None or locals().get('transfer') is not None, 'Unknown model'

    return (ds, epoch, value,
            locals().get('epochs'), locals().get('transfer', ''),
            locals().get('loss', 'BinaryCrossentropy'),
            locals().get('optimizer', 'Adam'),
            update_d({'lr': 1e-3}, optimizer_params),
            locals().get('base', 'transfer'))
