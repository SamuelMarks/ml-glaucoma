import gzip
import json
from random import sample
from functools import partial
from itertools import *
from collections import namedtuple
from os import path, remove, walk, getcwd, environ
from fnmatch import filter as fnmatch_filter
from socket import getfqdn

from PIL import Image

from openpyxl import load_workbook
from sas7bdat import SAS7BDAT

import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.base import Datasets
from tensorflow.contrib.learn.python.learn.datasets.mnist import _read32

from ml_glaucoma import logger
from ml_glaucoma.utils import run_once, it_consumes, pp, json_serial, redis_cursor
from ml_glaucoma.utils.cache import Cache

pickled_cache = {}
base_dir = None
just = 50
RecImg = namedtuple('RecImg', ('rec', 'imgs'))  # type: (generated_types.T0, [str])

cache = Cache(fname=environ.get('CACHE_FNAME') if 'NO_REDIS' in environ else redis_cursor)
rand_cache = Cache(fname=path.join(path.dirname(path.dirname(__file__)), '_data', '.cache', 'rand_cache.pkl')).load()
fqdn = getfqdn()


def _update_generated_types_py(args=None, replace=False):
    if path.isfile('generated_types.py'):
        if not replace:
            return
        else:
            remove('generated_types.py')
            remove('generated_types.pyc')

    with open('generated_types.py', 'wt') as f:
        f.write('from collections import namedtuple')
        f.write('\n\n')
        if args is None:
            f.write('T0 = None')
        else:
            f.write('T0 = namedtuple({T0_args[0]!r}, {T0_args[1]})'.format(T0_args=args))
        f.write('\n\n')


def _get_tbl(xlsx='glaucoma_20161205plus_Age23.xlsx'):
    global pickled_cache

    if 'tbl' in pickled_cache:
        return pickled_cache['tbl']

    wb = load_workbook(xlsx)
    rows_gen = wb.get_active_sheet().rows

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
        sas_tbl = dict(imap(lambda row: (row[0], RecImg(globals()['generated_types'].T0(*row), None)), f.readlines()))
        pickled_cache['sas_tbl'] = sas_tbl
        skip_save or cache.save(pickled_cache)

    return sas_tbl


def _imgs_of(idnum, img_directory):
    assert idnum.isdigit()
    return (path.join(root, filename)
            for root, dirnames, filenames in walk(img_directory)
            for filename in fnmatch_filter(filenames,
                                           '*{idnum}*'.format(idnum=idnum)
                                           # if fqdn == 'kudu' else '*{idnum}*10pc*'.format(idnum=idnum)
                                           )
            if '10pc' not in filename  # or fqdn != 'kudu'
            )


def _populate_imgs(img_directory, skip_save=True):
    global pickled_cache
    cache.update_locals(pickled_cache, locals())

    tbl = pickled_cache['tbl']

    if 'id_to_imgs' not in pickled_cache:
        if not path.isdir(img_directory):
            raise OSError('{} must exist and contain the images'.format(img_directory))

        pickled_cache['id_to_imgs'] = id_to_imgs = {e.rec.IDNUM: tuple(_imgs_of(e.rec.IDNUM, img_directory))
                                                    for e in tbl.itervalues()}
        pickled_cache['tbl'] = tbl = {id_: RecImg(recimg.rec, id_to_imgs[id_]) for id_, recimg in tbl.iteritems()}

    if 'imgs_to_id' not in pickled_cache:
        pickled_cache['imgs_to_id'] = imgs_to_id = {img: id_ for id_, imgs in id_to_imgs.iteritems()
                                                    for img in imgs}

    if 'total_imgs_assoc_to_id' not in pickled_cache:
        pickled_cache['total_imgs_assoc_to_id'] = total_imgs_assoc_to_id = len(imgs_to_id)

    if 'total_imgs' not in pickled_cache:
        pickled_cache['total_imgs'] = total_imgs = sum(
            len(filenames)
            for root, dirnames, filenames in walk(path.join('BMES123', 'BMES1Images'))
        )

    # Arghh, where are my views/slices?
    if not skip_save:
        print 'saving in _populate_imgs'
        cache.save(pickled_cache)

    logger.info('total_imgs == total_imgs_assoc_to_id:'.ljust(just) + '{}'.format(
        pickled_cache['total_imgs'] == pickled_cache['total_imgs_assoc_to_id']))

    return pickled_cache


def _vanilla_stats(skip_save=True):
    global pickled_cache
    cache.update_locals(pickled_cache, locals())

    tbl = pickled_cache['tbl']
    sas_tbl = pickled_cache['sas_tbl']

    if 'oags1' not in pickled_cache:
        pickled_cache['oags1'] = oags1 = tuple(v.rec.IDNUM for v in tbl.itervalues() if v.rec.oag1)
    else:
        oags1 = pickled_cache['oags1']  # Weird, this doesn't get into locals() from `update_locals`

    if 'no_oags1' not in pickled_cache:
        pickled_cache['no_oags1'] = no_oags1 = tuple(v.rec.IDNUM for v in tbl.itervalues() if not v.rec.oag1)

    if '_vanilla_stats' not in pickled_cache:
        pickled_cache['_vanilla_stats'] = vanilla_stats = '\n'.join(
            '{0}{1}'.format(*t) for t in (('# total:'.ljust(just), len(tbl)),
                                          ('# with oag1:'.ljust(just), len(oags1)),
                                          ('# with oag1 and roag1 and loag1:'.ljust(just),
                                           sum(1 for v in tbl.itervalues() if
                                               v.rec.oag1 and v.rec.roag1 and v.rec.loag1)),
                                          ('# with oag1 and roag1 and loag1 and glaucoma4:'.ljust(just),
                                           sum(1 for v in tbl.itervalues()
                                               if v.rec.oag1 and v.rec.roag1 and v.rec.loag1 and v.rec.glaucoma4)),
                                          ('# len(sas_tbl) == len(tbl):'.ljust(just),
                                           len(sas_tbl) == len(tbl)))
        )
        skip_save or cache.save(pickled_cache)

    it_consumes(imap(logger.info, pickled_cache['_vanilla_stats'].split('\n')))
    logger.info('oags1:'.ljust(just) + '{}'.format(oags1))
    logger.info('generated_types.T0._fields:'.ljust(just) + '{}'.format(generated_types.T0._fields))
    return pickled_cache


def get_datasets(skip_save=True):
    global pickled_cache
    get_data(skip_save=skip_save)

    no_oags1 = pickled_cache['no_oags1']
    oags1 = pickled_cache['oags1']
    tbl_ids = pickled_cache['tbl_ids']
    train, test = (frozenset(chain(
        (no_oags1[k] for k in rand_cache['2000 in 0-3547'][i * 970:970 + i * 970]),
        (oags1[k] for k in rand_cache['0-108'][i * 30:30 + i * 30])
    )) for i in xrange(2))

    pickled_cache['datasets'] = Datasets(train=train, test=test, validation=(train | test) ^ tbl_ids)
    return pickled_cache['datasets']


def _log_set_stats():
    global pickled_cache

    tbl = pickled_cache['tbl']
    datasets = pickled_cache['datasets']

    logger.info('# in train set:'.ljust(just) + str(len(datasets.train)))
    logger.info('# in test set:'.ljust(just) + str(len(datasets.test)))
    logger.info('# in validation set:'.ljust(just) + str(len(datasets.validation)))
    logger.info('# shared between sets:'.ljust(just) + str(sum((len(datasets.validation & datasets.test),
                                                                len(datasets.test & datasets.train),
                                                                len(datasets.validation & datasets.train)))))
    # '# shared between sets:'.ljust(just), sum(len(s0&s1) for s0, s1 in combinations((validation, test, train), 2))
    logger.info(
        '# len(all sets):'.ljust(just) + str(sum(imap(len, (datasets.train, datasets.test, datasets.validation)))))
    logger.info('# len(total):'.ljust(just) + str(len(tbl)))
    logger.info(
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


Data = namedtuple('Data', ('tbl', 'datasets', 'features', 'feature_names', 'pickled_cache'))


@run_once
def get_data(skip_save=True, cache_fname=None, invalidate=False):  # still saves once
    """
    Gets and optionally caches data, using SAS | XLSX files as index, and BMES root as files

    :keyword skip_save: Skips saving
    :type skip_save: ``bool``

    :keyword cache_fname: Cache filename. Defaults to $CACHE_FNAME
    :type cache_fname: ``str``

    :return: data
    :rtype: ``Data``
    """
    global pickled_cache
    global cache
    if cache_fname is not None:
        cache = Cache(fname=cache_fname)
    if invalidate:
        cache.invalidate()
    pickled_cache = cache.update_locals(cache.load(), locals()) if path.getsize('generated_types.py') > 50 else {}

    if pickled_cache:
        logger.info('imported T0 has:'.ljust(just) + '{}'.format(
            generated_types.T0._fields if generated_types.T0 is not None else generated_types.T0))

    global base_dir
    base_dir = base_dir or path.join(path.expanduser('~'), 'repos', 'thesis', 'BMES')
    _get_tbl(path.join(base_dir, 'glaucoma_20161205plus_Age23.xlsx'))
    _get_sas_tbl(path.join(base_dir, 'glaucoma_20161205plus_age23.sas7bdat'))

    _populate_imgs(img_directory=path.join(base_dir, 'BMES123', 'BMES1Images'), skip_save=skip_save)
    _vanilla_stats(skip_save=skip_save)

    datasets = get_datasets(skip_save=skip_save)
    _log_set_stats()

    feature_names = get_feature_names()
    features = get_features(feature_names, skip_save=skip_save)

    logger.info('feature_names:'.ljust(just) + '{}'.format(feature_names))
    logger.info('features:'.ljust(just) + '{}'.format(features))
    cache.save(pickled_cache)

    tbl = pickled_cache['tbl']
    sample = random_sample(tbl, datasets.train)
    pp(sample)
    # print 'datasets.train =', datasets.train
    # print 'tbl =', tbl
    # oags1 = pickled_cache['oags1']
    # for pid in oags1:
    if 'id_to_img_dims' not in pickled_cache:
        id_to_img_dims = {}
        img_dims_to_recimg = {10077696: set(), 48769206: set(), 487350: set(), 100751: set()}
        for IDNUM, u in tbl.iteritems():
            id_to_img_dims[u.rec.IDNUM] = set()
            for img_fname in u.imgs:
                with Image.open(img_fname) as img:
                    width, height = img.size

                dim = width * height
                id_to_img_dims[u.rec.IDNUM].add(dim)
                img_dims_to_recimg[dim].add(u)
                # print '{} * {} = {} is {}'.format(width, height, img.size, img_fname)
        pickled_cache['id_to_img_dims'] = id_to_img_dims
        pickled_cache['img_dims_to_recimg'] = img_dims_to_recimg
        cache.save(pickled_cache)
    else:
        id_to_img_dims = pickled_cache['id_to_img_dims']
        img_dims_to_recimg = pickled_cache['img_dims_to_recimg']

    logger.info('len(id_to_img_dims):'.ljust(just) + '{:d}'.format(len(id_to_img_dims)))
    logger.info('len(img_dims_to_recimg):'.ljust(just) + '{:d}'.format(len(img_dims_to_recimg)))
    logger.info('# without oags1 but with loag1 etc.:'.ljust(just) + '{:d}'.format(
        sum(1 for IDNUM, u in tbl.iteritems() if not u.rec.oag1 and (u.rec.roag1 or u.rec.loag1))))
    for dim in img_dims_to_recimg:
        logger.info(
            'len(img_dims_to_recimg[{:d}]):'.format(dim).ljust(just) + '{:d}'.format(len(img_dims_to_recimg[dim])))
        logger.info('{:d} with oags1:'.format(dim).ljust(just) + '{:d}'.format(
            sum(1 for recimg in img_dims_to_recimg[dim] if recimg.rec.oag1)))
        logger.info('{:d} with oags1:'.format(dim).ljust(just) + '{:d}'.format(
            sum(1 for recimg in img_dims_to_recimg[dim] if recimg.rec.oag1)))

    return Data(tbl, datasets, features, feature_names, pickled_cache)


def prepare_data(data_obj):
    def extract_images(filename):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        print 'Extracting', filename
        with gzip.open(filename) as bytestream:
            magic = _read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number %d in glaucoma image file: %s' % (magic, filename))
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols, 1)
            return data

    print random_sample(data_obj.tbl, data_obj.datasets.train)
    print 'data_obj.datasets.train =', data_obj.datasets.train
    return it_consumes(imap(lambda recimg: it_consumes(imap(extract_images, recimg.imgs)), data_obj.tbl.itervalues()))


_update_generated_types_py()
import generated_types

if __name__ == '__main__':
    _data = get_data()
    prepare_data(_data)
