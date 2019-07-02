from __future__ import print_function

import struct
from collections import namedtuple
from fnmatch import filter as fnmatch_filter
from functools import partial
from itertools import chain, groupby
from operator import itemgetter
from os import path, remove, walk, environ, symlink, makedirs
from platform import python_version_tuple
from random import sample
from shutil import rmtree
from socket import getfqdn
from sys import modules

import quantumrandom
from six import iteritems, itervalues

if python_version_tuple()[0] == '3':
    from importlib import reload
    from functools import reduce

    xrange = range
    imap = map
    ifilter = filter
    izip = zip
else:
    from itertools import imap, ifilter, izip

from openpyxl import load_workbook
from sas7bdat import SAS7BDAT

import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets

from ml_glaucoma import get_logger
from ml_glaucoma.utils import run_once, it_consumes, pp, redis_cursor
from ml_glaucoma.utils.cache import Cache

logger = get_logger(modules[__name__].__name__)
# logger.setLevel(CRITICAL)

pickled_cache = {}
just = 50
RecImg = namedtuple('RecImg', ('rec', 'imgs'))  # type: (generated_types.T0, [str])
globals()[RecImg.__name__] = RecImg

IdEyeFname = namedtuple('IdEyeFname', ('id', 'eye', 'fname'))

cache = Cache(fname=environ.get('CACHE_FNAME') if 'NO_REDIS' in environ else redis_cursor)
rand_cache_obj = Cache(fname=path.join(path.dirname(path.dirname(__file__)), '_data', '.cache', 'rand_cache.pkl'))
rand_cache_recreate = environ.get('RECREATE_RAND_CACHE', False)
rand_cache = rand_cache_obj.load()
fqdn = getfqdn()


def create_random_numbers(minimum, maximum, n):
    whole, prev = frozenset(), frozenset()
    while len(whole) < n:
        whole = reduce(frozenset.union,
                       (frozenset(imap(lambda num: minimum + (num % maximum),
                                       quantumrandom.get_data(data_type='uint16', array_length=1024))),
                        prev))
        prev = whole
        print(len(whole), 'of', n)
    return sample(whole, n)


class UnknownImageFormat(Exception):
    pass


def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core
    """
    size = path.getsize(file_path)

    with open(file_path) as f:
        height = -1
        width = -1
        data = f.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack('<HH', data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack('>LL', data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack('>LL', data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = ' raised while trying to decode as JPEG.'
            f.seek(0)
            f.read(2)
            b = f.read(1)
            try:
                while b and ord(b) != 0xDA:
                    while ord(b) != 0xFF: b = f.read(1)
                    while ord(b) == 0xFF: b = f.read(1)
                    if 0xC0 <= ord(b) <= 0xC3:
                        f.read(3)
                        h, w = struct.unpack('>HH', f.read(4))
                        break
                    else:
                        f.read(int(struct.unpack('>H', f.read(2))[0]) - 2)
                    b = f.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat('StructError' + msg)
            except ValueError:
                raise UnknownImageFormat('ValueError' + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat('Sorry, don\'t know how to get information from this file.')

        return width, height


def _update_generated_types_py(args=None, replace=False):
    gen_py = 'generated_types.py'
    if path.isfile(gen_py):
        if not replace:
            return
        else:
            remove(gen_py)
            gen_pyc = '{}c'.format(gen_py)
            if path.isfile(gen_pyc):
                remove(gen_pyc)

    with open('generated_types.py', 'wt') as f:
        f.write('from collections import namedtuple')
        f.write('\n\n')
        # f.write("RecImg = namedtuple('RecImg', ('rec', 'imgs'))\n")
        if args is None:
            f.write('T0 = None')
        else:
            f.write('T0 = namedtuple({T0_args[0]!r}, {T0_args[1]})\n'.format(T0_args=args))
        f.write('\n\n')


def _get_tbl(xlsx='glaucoma_20161205plus_Age23.xlsx'):
    global pickled_cache

    if 'tbl' in pickled_cache:
        return pickled_cache['tbl']

    wb = load_workbook(xlsx)
    rows_gen = wb.active.rows

    get_vals = partial(imap, lambda col: col.value)

    if generated_types.T0:
        logger.warn('skipping header')
        next(rows_gen)  # skip header
    else:
        T0_args = ('T0', tuple(get_vals(next(rows_gen))))
        _update_generated_types_py(T0_args, replace=True)
        reload(generated_types)

    tbl = dict(imap(lambda row: (row[0].value,
                                 RecImg(globals()['generated_types'].T0(*get_vals(row)), None)), rows_gen))
    pickled_cache['tbl'] = tbl  # type: (str, get_data.RecImg)
    pickled_cache['tbl_ids'] = frozenset(tbl.keys())
    return tbl


def _get_sas_tbl(sas7bdat='glaucoma_20161205plus_age23.sas7bdat', skip_save=True):
    global pickled_cache

    if 'sas_tbl' in pickled_cache:
        return pickled_cache['sas_tbl']

    with SAS7BDAT(sas7bdat, skip_header=True) as f:
        sas_tbl = dict(
            imap(lambda row: (row[0], RecImg(globals()['generated_types'].T0(*row), None)), f.readlines()))
        pickled_cache['sas_tbl'] = sas_tbl
        skip_save or cache.save(pickled_cache)

    return sas_tbl


def _populate_imgs(img_directory, skip_save=True):
    global pickled_cache
    cache.update_locals(pickled_cache, locals())

    tbl = pickled_cache['tbl']

    if 'id2ideyefname' not in pickled_cache:
        if not path.isdir(img_directory):
            raise OSError('{} must exist and contain the images'.format(img_directory))

        pickled_cache['all_imgs'] = all_imgs = tuple(sorted(
            (IdEyeFname(*(lambda fname: (fname[:-1], fname[-1], path.join(root, filename)))(
                filename[filename.rfind('BMES') + len('BMES') + 1:filename.rfind('-')].partition('-')[0]))
             for root, dirnames, filenames in walk(img_directory)
             for filename in fnmatch_filter(filenames, '*.jpg'))
            , key=itemgetter(0)))  # type: tuple(IdEyeFname)

        pickled_cache['id2ideyefname'] = id2ideyefname = {key: d[key] for d in tuple(
            {_id: tuple(group)} for _id, group in groupby(all_imgs, key=itemgetter(0))
        ) for key in d}  # type: {str: tuple(IdEyeFname)}

        pickled_cache['total_imgs_assoc_to_id'] = total_imgs_assoc_to_id = sum(
            len(v) for k, v in iteritems(id2ideyefname))
        pickled_cache['tbl'] = tbl = {id_: RecImg(recimg.rec, id2ideyefname[id_])
                                      for id_, recimg in iteritems(tbl)
                                      if id_ in id2ideyefname}

    else:
        id2ideyefname = pickled_cache['id2ideyefname']
        all_imgs = pickled_cache['all_imgs']
        total_imgs_assoc_to_id = pickled_cache['total_imgs_assoc_to_id']

    if 'imgs_to_id' not in pickled_cache:
        pickled_cache['imgs_to_id'] = imgs_to_id = {ideyefname.fname: id_
                                                    for id_, ideyefnames in iteritems(id2ideyefname)
                                                    for ideyefname in ideyefnames}
    else:
        imgs_to_id = pickled_cache['imgs_to_id']

    if 'total_imgs' not in pickled_cache:
        pickled_cache['total_imgs'] = total_imgs = sum(
            sum(1 for fname in filenames if fname.endswith('.jpg'))
            for root, dirnames, filenames in walk(img_directory)
        )
    else:
        total_imgs = pickled_cache['total_imgs']

    # Arghh, where are my views/slices?
    if not skip_save:
        logger.debug('saving in _populate_imgs')
        cache.save(pickled_cache)

    logger.debug('total_imgs == len(all_imgs) == len(imgs_to_id):'.ljust(just) + '{}'.format(
        total_imgs == len(all_imgs) == len(imgs_to_id)
    ))
    logger.debug('# not allocated in tbl:'.ljust(just) + '{}'.format(len(id2ideyefname) - len(tbl)))

    return pickled_cache


def _vanilla_stats(skip_save=True):
    global pickled_cache
    cache.update_locals(pickled_cache, locals())

    tbl = pickled_cache['tbl']
    sas_tbl = pickled_cache['sas_tbl']
    id2ideyefname = pickled_cache['id2ideyefname']

    if 'oags1' not in pickled_cache:
        pickled_cache['oags1'] = oags1 = tuple(v.rec.IDNUM for v in itervalues(tbl) if v.rec.oag1)
    else:
        oags1 = pickled_cache['oags1']  # Weird, this doesn't get into locals() from `update_locals`

    if 'loags1' not in pickled_cache:
        pickled_cache['loag1'] = loag1 = tuple(v.rec.IDNUM for v in itervalues(tbl) if v.rec.loag1)
        pickled_cache['roag1'] = roag1 = tuple(v.rec.IDNUM for v in itervalues(tbl) if v.rec.roag1)
    else:
        loag1 = pickled_cache['loag1']
        roag1 = pickled_cache['roag1']

    if 'no_oags1' not in pickled_cache:
        pickled_cache['no_oags1'] = no_oags1 = tuple(v.rec.IDNUM for v in itervalues(tbl) if not v.rec.oag1)

    if '_vanilla_stats' not in pickled_cache:
        pickled_cache['_vanilla_stats'] = vanilla_stats = '\n'.join(
            '{0}{1}'.format(*t) for t in (('# total:'.ljust(just), len(tbl)),
                                          ('# with oag1:'.ljust(just), len(oags1)),
                                          ('# with roag1:'.ljust(just), len(roag1)),
                                          ('# with loag1:'.ljust(just), len(loag1)),
                                          ('# with oag1 and roag1 and loag1:'.ljust(just),
                                           sum(1 for v in itervalues(tbl) if
                                               v.rec.oag1 and v.rec.roag1 and v.rec.loag1)),
                                          ('# with oag1 and roag1 and loag1 and glaucoma4:'.ljust(just),
                                           sum(1 for v in itervalues(tbl)
                                               if v.rec.oag1 and v.rec.roag1 and v.rec.loag1 and v.rec.glaucoma4)),
                                          ('# len(sas_tbl) == len(tbl):'.ljust(just),
                                           len(sas_tbl) == len(tbl)))
        )
        skip_save or cache.save(pickled_cache)

    it_consumes(imap(logger.debug, pickled_cache['_vanilla_stats'].split('\n')))
    logger.debug('oags1:'.ljust(just) + '{}'.format(oags1))
    logger.debug('loag1:'.ljust(just) + '{}'.format(loag1))

    if 'loags_id2fname' not in pickled_cache:
        id2fname = lambda dataset, eye: (lambda l: dict(izip(l[::2], l[1::2])))(tuple(chain.from_iterable(
            imap(lambda idnum_group: (idnum_group[0], tuple(imap(itemgetter(1), idnum_group[1]))),
                 groupby(
                     chain.from_iterable(
                         imap(lambda ideyefnames: tuple(
                             imap(lambda ideyefname: (ideyefname.id, ideyefname.fname), ideyefnames)),
                              imap(lambda ideyefnames: ifilter(lambda ideyefname: ideyefname.eye == eye, ideyefnames),
                                   imap(lambda idnum: id2ideyefname[idnum], dataset)))), key=itemgetter(0)
                 )))))

        pickled_cache['loags_id2fname'] = loags_id2fname = id2fname(dataset=loag1, eye='L')
        pickled_cache['roags_id2fname'] = roags_id2fname = id2fname(dataset=roag1, eye='R')
    else:
        loags_id2fname = pickled_cache['loags_id2fname']
        roags_id2fname = pickled_cache['roags_id2fname']
    # pp(loags_id2fname)

    logger.debug('generated_types.T0._fields:'.ljust(just) + '{}'.format(generated_types.T0._fields))
    return pickled_cache


def make_symlinks(dest_dir, filenames, clean_dir=False):
    if path.isdir(dest_dir):
        if clean_dir:
            rmtree(dest_dir)
            makedirs(dest_dir)  # no goto :(
    else:
        makedirs(dest_dir)

    it_consumes(imap(lambda fname: symlink(fname, path.join(dest_dir, path.basename(fname))), filenames))


def get_datasets(no_oags=970, oags=30, skip_save=True):
    """
    Partitions datasets into train, test, and validation

    :keyword no_oags: Number of Open Angle Glaucoma negative to include in test [#no_oags] and train [#no_oags].
    Everything leftover goes into validation.
    :type no_oags: ``int``

    :keyword oags: Number of Open Angle Glaucoma positive to include in test [#no_oags] and train [#no_oags].
    Everything leftover goes into validation.
    :type oags: ``int``

    :keyword skip_save: Skips saving
    :type skip_save: ``bool``

    :return: Partitioned dataset (a namedtuple)
    :rtype ``Datasets``
    """
    global pickled_cache
    get_data(skip_save=skip_save)

    no_oags1 = pickled_cache['no_oags1']
    oags1 = pickled_cache['oags1']
    tbl_ids = pickled_cache['tbl_ids']
    # rand_cache = pickled_cache['rand_cache']
    train, test = (frozenset(chain(
        (no_oags1[k] for k in rand_cache['0-13661'][i * no_oags:no_oags + i * no_oags]),
        (oags1[k] for k in rand_cache['0-412'][i * oags:oags + i * oags])
    )) for i in xrange(2))

    pickled_cache['datasets'] = Datasets(train=train, test=test, validation=(train | test) ^ tbl_ids)
    return pickled_cache['datasets']


def _log_set_stats():
    global pickled_cache

    tbl = pickled_cache['tbl']
    datasets = pickled_cache['datasets']

    logger.debug('# in train set:'.ljust(just) + str(len(datasets.train)))
    logger.debug('# in test set:'.ljust(just) + str(len(datasets.test)))
    logger.debug('# in validation set:'.ljust(just) + str(len(datasets.validation)))
    logger.debug('# shared between sets:'.ljust(just) + str(sum((len(datasets.validation & datasets.test),
                                                                 len(datasets.test & datasets.train),
                                                                 len(datasets.validation & datasets.train)))))
    # '# shared between sets:'.ljust(just), sum(len(s0&s1) for s0, s1 in combinations((validation, test, train), 2))
    logger.debug(
        '# len(all sets):'.ljust(just) + str(sum(imap(len, (datasets.train, datasets.test, datasets.validation)))))
    logger.debug('# len(total):'.ljust(just) + str(len(tbl)))
    logger.debug(
        '# len(all sets) == len(total):'.ljust(just) + str(
            sum(imap(len, (datasets.train, datasets.test, datasets.validation))) == len(tbl)))


def random_sample(tbl, ids, num=1):
    if num != 1:
        raise NotImplementedError()
    return tbl[sample(ids, num)[0]]


def get_feature_names():
    return tuple(field for field in generated_types.T0._fields
                 if field.endswith('1'))  # BMES1


def get_features(feature_names, skip_save=True):
    return np.bmat(np.fromiter((idx for idx, _ in enumerate(feature_names)), np.float32))


# Random dead code?
'''
train = _create_dataset(data_obj, data_obj.datasets.train)
validation = _create_dataset(data_obj, data_obj.datasets.validation)
test = _create_dataset(data_obj, data_obj.datasets.test)
return train, validation, test
'''

Data = namedtuple('Data', ('tbl', 'datasets', 'features', 'feature_names', 'pickled_cache'))


def link_distribute_dataset(train_ratio, test_ratio, train_positive_ratio, test_positive_ratio, split_dir,
                            max_imgs=None):
    global rand_cache, pickled_cache

    train_dir = path.join(split_dir, 'train')
    test_dir = path.join(split_dir, 'test')
    valid_dir = path.join(split_dir, 'valid')

    glaucoma_fnames = tuple(pickled_cache['glaucoma_fnames'])
    bmes1_no_glaucoma_fnames = tuple(pickled_cache['bmes1_no_glaucoma_fnames'])

    len_glaucoma_fnames = len(glaucoma_fnames)
    len_no_glaucoma_fnames = len(bmes1_no_glaucoma_fnames)

    rand_glaucoma = rand_cache['0-{:d}'.format(len_glaucoma_fnames)]
    rand_no_glaucoma = rand_cache['0-{:d}'.format(len_no_glaucoma_fnames)]

    outer_locals = locals()

    def n_stats(dataset):
        _no_glaucoma_n = int(outer_locals['{}_ratio'.format(dataset)] * len_no_glaucoma_fnames)
        _glaucoma_n = int(outer_locals['{}_positive_ratio'.format(dataset)] * len_glaucoma_fnames)
        logger.debug('# of negative {} images:'.format(dataset).ljust(just) + '{:d}'.format(_no_glaucoma_n))
        logger.debug('# of positive {} images:'.format(dataset).ljust(just) + '{:d}'.format(_glaucoma_n))
        return _no_glaucoma_n, _glaucoma_n

    train_no_glaucoma_n, train_glaucoma_n = n_stats('train')
    test_no_glaucoma_n, test_glaucoma_n = n_stats('test')

    def idx_to_tuple(d, indices):
        return tuple(imap(lambda idx: d[idx], indices))

    # ACTION!

    train_glaucoma_fnames = idx_to_tuple(glaucoma_fnames, rand_glaucoma[:train_glaucoma_n])
    train_no_glaucoma_fnames = idx_to_tuple(bmes1_no_glaucoma_fnames, rand_no_glaucoma[:train_no_glaucoma_n])
    if max_imgs:
        train_glaucoma_fnames = train_glaucoma_fnames[:max_imgs]
        train_no_glaucoma_fnames = train_no_glaucoma_fnames[:max_imgs]

    make_symlinks(path.join(train_dir, 'glaucoma'), filenames=train_glaucoma_fnames, clean_dir=True)
    make_symlinks(path.join(train_dir, 'no_glaucoma'), filenames=train_no_glaucoma_fnames, clean_dir=True)

    test_glaucoma_fnames = idx_to_tuple(glaucoma_fnames,
                                        rand_glaucoma[train_glaucoma_n:train_glaucoma_n + test_glaucoma_n])
    test_no_glaucoma_fnames = idx_to_tuple(
        bmes1_no_glaucoma_fnames,
        rand_no_glaucoma[train_no_glaucoma_n:train_no_glaucoma_n + test_no_glaucoma_n]
    )

    if max_imgs:
        test_glaucoma_fnames = test_glaucoma_fnames[:max_imgs]
        test_no_glaucoma_fnames = test_no_glaucoma_fnames[:max_imgs]

    make_symlinks(path.join(test_dir, 'glaucoma'), filenames=test_glaucoma_fnames, clean_dir=True)
    make_symlinks(path.join(test_dir, 'no_glaucoma'), filenames=test_no_glaucoma_fnames, clean_dir=True)

    valid_glaucoma_fnames = idx_to_tuple(glaucoma_fnames, rand_glaucoma[train_glaucoma_n + test_glaucoma_n:])
    valid_no_glaucoma_fnames = idx_to_tuple(bmes1_no_glaucoma_fnames,
                                            rand_no_glaucoma[train_no_glaucoma_n + test_no_glaucoma_n:])

    if max_imgs:
        valid_glaucoma_fnames = valid_glaucoma_fnames[:max_imgs]
        valid_no_glaucoma_fnames = valid_no_glaucoma_fnames[:max_imgs]

    make_symlinks(path.join(valid_dir, 'glaucoma'), filenames=valid_glaucoma_fnames, clean_dir=True)
    make_symlinks(path.join(valid_dir, 'no_glaucoma'), filenames=valid_no_glaucoma_fnames, clean_dir=True)

    return Datasets(train=train_dir, test=test_dir, validation=valid_dir)


@run_once
def get_data(base_dir, split_dir,
             train_ratio=.8, test_ratio=.1, train_positive_ratio=.8, test_positive_ratio=.1,
             skip_save=True, cache_fname=None, max_imgs=None,
             invalidate=False):  # still saves once
    """
    Gets and optionally caches data, using SAS | XLSX files as index, and BMES root as files

    :keyword base_dir: Base dir. Should have a BMES123 folder inside.
    :type base_dir: ``str``

    :keyword split_dir: Directory to place parsed files. NOTE: These are symbolically linked from the base dir.
    :type split_dir: ``str``

    :keyword train_ratio: Proportion of the dataset to include in the test split.
    By default, the value is set to 0.8. Everything leftover goes into validation.
    :type train_ratio: ``float``

    :keyword test_ratio: Proportion of the dataset to include in the train split.
    By default, the value is set to 0.1. Everything leftover goes into validation.
    :type test_ratio: ``float``

    :keyword train_positive_ratio: Proportion of the glaucoma-present dataset to include in the train split.
    By default, the value is set to 0.5. Everything leftover goes into validation.
    :type train_positive_ratio: ``float``

    :keyword test_positive_ratio: Proportion of the glaucoma-present dataset to include in the test split.
    By default, the value is set to 0.5. Everything leftover goes into validation.
    :type test_positive_ratio: ``float``

    :keyword skip_save: Skips saving
    :type skip_save: ``bool``

    :keyword cache_fname: Cache filename. Defaults to $CACHE_FNAME
    :type cache_fname: ``str``

    :keyword max_imgs: Maximum number of images to process; `None` will process all images
    :type max_imgs: ``int``

    :keyword invalidate: Invalidate cache first
    :type invalidate: ``bool``

    :return: Datasets (directory strings, subdirs of 'glaucoma', 'no_glaucoma')
    :rtype: ``Datasets``
    """
    global pickled_cache
    global cache
    if cache_fname is not None:
        cache = Cache(fname=cache_fname)
    if invalidate:
        cache.invalidate()
    pickled_cache = cache.update_locals(cache.load(), locals()) if path.getsize('generated_types.py') > 50 else {}

    if pickled_cache:
        logger.debug('imported T0 has:'.ljust(just) + '{}'.format(
            generated_types.T0._fields if generated_types.T0 is not None else generated_types.T0))

    assert base_dir, 'No database directory provided'
    _get_tbl(path.join(base_dir, 'glaucoma_20161205plus_Age23.xlsx'))
    _get_sas_tbl(path.join(base_dir, 'glaucoma_20161205plus_age23.sas7bdat'))

    _populate_imgs(img_directory=path.join(base_dir, 'BMES123'), skip_save=skip_save)
    _vanilla_stats(skip_save=skip_save)

    if 'glaucoma_fnames' not in pickled_cache:
        all_imgs = pickled_cache['all_imgs']
        loags_id2fname = pickled_cache['loags_id2fname']
        roags_id2fname = pickled_cache['roags_id2fname']
        pickled_cache['glaucoma_fnames'] = glaucoma_fnames = frozenset(
            chain.from_iterable(chain.from_iterable((itervalues(loags_id2fname), itervalues(roags_id2fname))))
        )
        pickled_cache['bmes1_no_glaucoma_fnames'] = bmes1_no_glaucoma_fnames = frozenset(
            imap(lambda ideyefname: ideyefname.fname, ifilter(
                lambda ideyefname: path.basename(
                    path.dirname(path.dirname(path.dirname(ideyefname.fname)))) == 'BMES1Images', all_imgs
            ))) - glaucoma_fnames
    else:
        glaucoma_fnames = pickled_cache['glaucoma_fnames']
        bmes1_no_glaucoma_fnames = pickled_cache['bmes1_no_glaucoma_fnames']

    global rand_cache_recreate, rand_cache, rand_cache_obj
    if rand_cache_recreate:
        rand_cache = {}
        len_glaucoma_fnames = len(glaucoma_fnames)
        len_no_glaucoma_fnames = len(bmes1_no_glaucoma_fnames)
        n = len(glaucoma_fnames) + len(bmes1_no_glaucoma_fnames)
        rand_cache['0-{:d}'.format(len_glaucoma_fnames)] = create_random_numbers(
            n=len_glaucoma_fnames, minimum=0, maximum=len_glaucoma_fnames
        )
        rand_cache['0-{:d}'.format(len_no_glaucoma_fnames)] = create_random_numbers(
            n=len_no_glaucoma_fnames, minimum=0, maximum=len_no_glaucoma_fnames
        )
        rand_cache_obj.save(rand_cache)

    return link_distribute_dataset(train_ratio=train_ratio, test_ratio=test_ratio,
                                   train_positive_ratio=train_positive_ratio, test_positive_ratio=test_positive_ratio,
                                   split_dir=split_dir, max_imgs=max_imgs)


def old(no_oags, oags, skip_save):
    datasets = get_datasets(no_oags=no_oags, oags=oags, skip_save=skip_save)
    _log_set_stats()

    feature_names = get_feature_names()
    features = get_features(feature_names, skip_save=skip_save)

    logger.debug('feature_names:'.ljust(just) + '{}'.format(feature_names))
    logger.debug('features:'.ljust(just) + '{}'.format(features))
    cache.save(pickled_cache)

    tbl = pickled_cache['tbl']
    curr_sample = random_sample(tbl, datasets.train)
    pp(curr_sample)
    # print 'datasets.train =', datasets.train
    # print 'tbl =', tbl
    # oags1 = pickled_cache['oags1']
    # for pid in oags1:
    if 'id_to_img_dims' not in pickled_cache:
        id_to_img_dims = {}
        img_dims_to_recimg = {10077696: set(), 48769206: set(), 487350: set(), 100751: set()}
        for IDNUM, u in iteritems(tbl):
            id_to_img_dims[u.rec.IDNUM] = set()
            for img_fname in u.imgs:
                # with Image.open(img_fname) as img: width, height = img.size
                width, height = get_image_size(img_fname)

                dim = width * height
                id_to_img_dims[u.rec.IDNUM].add(dim)
                logger.debug(
                    'dim:'.ljust(just) + '{dim} [{width} * {height}]'.format(width=width, height=height, dim=dim))
                img_dims_to_recimg[dim].add(u)
                # print '{} * {} = {} is {}'.format(width, height, img.size, img_fname)
        pickled_cache['id_to_img_dims'] = id_to_img_dims
        pickled_cache['img_dims_to_recimg'] = img_dims_to_recimg
        cache.save(pickled_cache)
    else:
        id_to_img_dims = pickled_cache['id_to_img_dims']
        img_dims_to_recimg = pickled_cache['img_dims_to_recimg']

    logger.debug('len(id_to_img_dims):'.ljust(just) + '{:d}'.format(len(id_to_img_dims)))
    logger.debug('len(img_dims_to_recimg):'.ljust(just) + '{:d}'.format(len(img_dims_to_recimg)))
    logger.debug('img_dims_to_recimg.keys():'.ljust(just) + '{}'.format(img_dims_to_recimg.keys()))
    logger.debug('img_dims_to_recimg:'.ljust(just) + '{}'.format(img_dims_to_recimg))
    logger.debug('# without oags1 but with loag1 etc.:'.ljust(just) + '{:d}'.format(
        sum(1 for IDNUM, u in iteritems(tbl) if not u.rec.oag1 and (u.rec.roag1 or u.rec.loag1))
    ))
    logger.debug('# with loag1:'.ljust(just) + '{:d}'.format(
        sum(1 for IDNUM, u in iteritems(tbl) if u.rec.loag1)
    ))
    logger.debug('# with roag1:'.ljust(just) + '{:d}'.format(
        sum(1 for IDNUM, u in iteritems(tbl) if u.rec.roag1)
    ))
    for dim in img_dims_to_recimg:
        logger.debug(
            'len(img_dims_to_recimg[{:d}]):'.format(dim).ljust(just) + '{:d}'.format(len(img_dims_to_recimg[dim])))
        logger.debug('{:d} with oags1:'.format(dim).ljust(just) + '{:d}'.format(
            sum(1 for recimg in img_dims_to_recimg[dim] if recimg.rec.oag1)))
        logger.debug('{:d} with oags1:'.format(dim).ljust(just) + '{:d}'.format(
            sum(1 for recimg in img_dims_to_recimg[dim] if recimg.rec.oag1)))

    data = Data(tbl, datasets, features, feature_names, pickled_cache)
    return data


_update_generated_types_py()
import generated_types

if __name__ == '__main__':
    _data = get_data(base_dir='/data', split_dir='/tmp/b')
    # train, val, test = prepare_data(_data, pixels=200)
    # print(train)

'''
/Users/samuel/repos/.venvs/thenv/bin/python /Users/samuel/repos/thesis/ml-glaucoma/ml_glaucoma/utils/get_data.py
2018-06-16 00:29:03,925 - get_data - WARNING - skipping header
2018-06-16 00:38:13,841 - get_data - DEBUG - total_imgs == total_imgs_assoc_to_id:             False
2018-06-16 00:38:13,866 - __init__ - DEBUG - # total:                                          3654
2018-06-16 00:38:13,866 - __init__ - DEBUG - # with oag1:                                      108
2018-06-16 00:38:13,866 - __init__ - DEBUG - # with oag1 and roag1 and loag1:                  52
2018-06-16 00:38:13,866 - __init__ - DEBUG - # with oag1 and roag1 and loag1 and glaucoma4:    5
2018-06-16 00:38:13,866 - __init__ - DEBUG - # len(sas_tbl) == len(tbl):                       True
2018-06-16 00:38:13,866 - get_data - DEBUG - oags1:                                            (u'0836', u'0749', u'0619', u'0422', u'0420', u'1758', u'1757', u'1756', u'1176', u'0447', u'2274', u'2695', u'1144', u'1542', u'0529', u'2194', u'2425', u'2517', u'2510', u'1817', u'2185', u'2189', u'2438', u'0088', u'1801', u'1948', u'1623', u'1622', u'0505', u'0148', u'0373', u'0656', u'0095', u'1485', u'1135', u'3018', u'1002', u'0799', u'1728', u'3003', u'1031', u'0782', u'0236', u'4038', u'0120', u'0535', u'2390', u'2500', u'2174', u'1738', u'0682', u'0118', u'0752', u'1307', u'0921', u'2575', u'2373', u'2391', u'0827', u'1260', u'4059', u'0594', u'0689', u'0105', u'1095', u'0106', u'2393', u'1966', u'3604', u'2359', u'3736', u'2135', u'1402', u'3892', u'1468', u'0481', u'0482', u'0483', u'1886', u'2651', u'1652', u'1651', u'3702', u'0365', u'3716', u'0635', u'2245', u'2240', u'2468', u'1188', u'0204', u'2668', u'3798', u'0801', u'0802', u'4010', u'0171', u'2499', u'3508', u'2206', u'3505', u'2674', u'3215', u'1638', u'2215', u'3370', u'1162', u'1566')
2018-06-16 00:38:13,866 - get_data - DEBUG - generated_types.T0._fields:                       ('IDNUM', 'age1', 'sex1', 'age4', 'sex4', 'r4vf1', 'l4vf1', 'Zeiss4', 'Canon4', 'oct4', 'bm4', 'Cpoint', 'SurvFlag', 'oag_dur', 'oag1', 'roag1', 'loag1', 'oag23inc', 'roag23inc', 'loag23inc', 'incglau_r4', 'result_r4', 'incglau_l4', 'result_l4', 'agegp1', 'glaucoma4', 'glaucoma4inc', 'bm4testing', 'AGE2', 'age3')
2018-06-16 00:38:13,867 - get_data - DEBUG - # in train set:                                   1000
2018-06-16 00:38:13,867 - get_data - DEBUG - # in test set:                                    1000
2018-06-16 00:38:13,867 - get_data - DEBUG - # in validation set:                              1654
2018-06-16 00:38:13,867 - get_data - DEBUG - # shared between sets:                            0
2018-06-16 00:38:13,867 - get_data - DEBUG - # len(all sets):                                  3654
2018-06-16 00:38:13,867 - get_data - DEBUG - # len(total):                                     3654
2018-06-16 00:38:13,867 - get_data - DEBUG - # len(all sets) == len(total):                    True
2018-06-16 00:38:13,867 - get_data - DEBUG - feature_names:                                    ('age1', 'sex1', 'r4vf1', 'l4vf1', 'oag1', 'roag1', 'loag1', 'agegp1')
2018-06-16 00:38:13,868 - get_data - DEBUG - features:                                         [[0. 1. 2. 3. 4. 5. 6. 7.]]
RecImg(rec=T0(IDNUM=u'0475', age1=64L, sex1=2L, age4=None, sex4=None, r4vf1=None, l4vf1=None, Zeiss4=None, Canon4=None, oct4=None, bm4=None, Cpoint=datetime.datetime(2007, 12, 31, 0, 0), SurvFlag=0L, oag_dur=10L, oag1=0L, roag1=0L, loag1=0L, oag23inc=0L, roag23inc=0L, loag23inc=0L, incglau_r4=None, result_r4=None, incglau_l4=None, result_l4=None, agegp1=2L, glaucoma4=None, glaucoma4inc=None, bm4testing=None, AGE2=71L, age3=76L), imgs=('/data/BMES123/BMES1Images/R-checked byLauren/BMES10400-10499R/BMES10475R-M.jpg', '/data/BMES123/BMES1Images/R-checked byLauren/BMES10400-10499R/BMES10475R-D.jpg', '/data/BMES123/BMES1Images/L-checkedbyLauren/BMES10400-10499L/BMES10475L-M.jpg', '/data/BMES123/BMES1Images/L-checkedbyLauren/BMES10400-10499L/BMES10475L-D.jpg'))
2018-06-16 00:38:25,638 - get_data - DEBUG - len(id_to_img_dims):                              3654
2018-06-16 00:38:25,638 - get_data - DEBUG - len(img_dims_to_recimg):                          4
2018-06-16 00:38:25,642 - get_data - DEBUG - # without oags1 but with loag1 etc.:              0
2018-06-16 00:38:25,642 - get_data - DEBUG - len(img_dims_to_recimg[10077696]):                3564
2018-06-16 00:38:25,644 - get_data - DEBUG - 10077696 with oags1:                              104
2018-06-16 00:38:25,645 - get_data - DEBUG - 10077696 with oags1:                              104
2018-06-16 00:38:25,645 - get_data - DEBUG - len(img_dims_to_recimg[487350]):                  0
2018-06-16 00:38:25,645 - get_data - DEBUG - 487350 with oags1:                                0
2018-06-16 00:38:25,645 - get_data - DEBUG - 487350 with oags1:                                0
2018-06-16 00:38:25,645 - get_data - DEBUG - len(img_dims_to_recimg[48769206]):                32
2018-06-16 00:38:25,645 - get_data - DEBUG - 48769206 with oags1:                              1
2018-06-16 00:38:25,645 - get_data - DEBUG - 48769206 with oags1:                              1
2018-06-16 00:38:25,646 - get_data - DEBUG - len(img_dims_to_recimg[100751]):                  0
2018-06-16 00:38:25,646 - get_data - DEBUG - 100751 with oags1:                                0
2018-06-16 00:38:25,646 - get_data - DEBUG - 100751 with oags1:                                0
(?, ?, 3)
(200, 200, 3)
(?, ?, 3)
(200, 200, 3)
(?, ?, 3)
(200, 200, 3)
<MapDataset shapes: ((200, 200, 3), ()), types: (tf.float32, tf.int32)>

Process finished with exit code 0

'''
