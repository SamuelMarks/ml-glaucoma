#!/usr/bin/env python
import sys
from functools import partial
from itertools import filterfalse, chain
from operator import itemgetter, contains
from os import path

import numpy as np
import pandas as pd


def isnotebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError, ImportError):
        return False  # Probably standard Python interpreter


if isnotebook():
    from IPython.display import display, HTML
else:
    display = print
    HTML = lambda ident: ident


    def new_p(*args, sep=''):
        s, prev = '<p>', ''
        for arg in args:
            s += '{}{}'.format('&#9;' if prev.rstrip().endswith(':') else '</p><p>',
                               arg)
            prev = arg
        s += '</p>'
        sys.stdout.write(s)
        sys.stdout.flush()


    print = new_p

from six import string_types


def chain_unique(*args):
    seen = set()
    yield from (v for v in chain(*args)
                if v not in seen and not seen.add(v))


image_angle_types = frozenset(('R1', 'R2', 'L1', 'L2'))

manycat2threecat = {
    'Maculopathy': (
        'non-referable',  # [0] No diabetic maculopathy
        'referable',  # [1] HEx distant from the fovea
        'referable',  # [2] HEx approaching the fovea
        'referable',  # [3] HEx involving the fovea
        'referable',  # [4] Maculopathy, unspecified
        'No gradable image'  # [5] No gradable image
    ),
    'ETDRS Grading': (
        'non-referable',  # [0] No DR
        'referable',  # [1] Mild non-proliferative (mild pre-proliferative)
        'referable',  # [2] Moderate non-proliferative/ moderate pre-proliferative
        'referable',  # [3] Severe non-proliferative/ severe pre-proliferative
        'referable',  # [4] Proliferative retinopathy
        'referable',  # [5] Pre-retinal fibrosis+/- tractional retinal detachment
        'referable',  # [6] Treated proliferative retinopathy, Unstable
        'referable',  # [7] Treated proliferative retinopathy, Stable
        'No gradable image'  # [8] No gradable image
    ),
    'Overall Findings': (
        np.nan,
        'referable',  # [1] Vision-threatening retinopathy
        'referable',  # [2] Non-proliferative diabetic retinopathy
        'non-referable',  # [3] No DR
        'No gradable image'  # [4] Ungradable
    ),
    'Overall Quality of the Photographs Taken': (
        np.nan,
        'No gradable image',  # [1] Inadequate for any diagnostic purpose
        'No gradable image',  # [2] Unable to exclude emergent findings
        'No gradable image',  # [3] Only able to exclude emergent findings
        'No gradable image',  # [4] Not ideal but still able to exclude subtle findings
        'referable',  # [5] Ideal quality
    )
}


def to_manycat_name(o):
    if isinstance(o, string_types):
        o = o,

    for e in o[::-1]:
        lower_e = e.lower()
        if lower_e == 'overall quality of the photographs taken':
            return 'Overall Quality of the Photographs Taken'
        elif e.startswith('ETDRS') or e == 'Overall Findings':
            return e
        elif 'macul' in lower_e:
            # print('matched with: {!r}'.format(e))
            return 'Maculopathy'
        elif e.startswith('Overall Finding'):
            return 'Overall Findings'
        else:
            print('no match found for: {!r}'.format(e))

    raise TypeError('{!r} no key found for'.format(o))


def grad_mac2(series):
    def from_s(value):
        if pd.isnull(value) or isinstance(value, string_types):
            return value
        value = np.ushort(value)
        name = series.name if series.name in manycat2threecat else to_manycat_name(series.name)

        mapped = manycat2threecat.get(name)

        return value if mapped is None or len(mapped) < value else mapped[value]

    return series if series is None else series.apply(from_s)


def debug(obj, name='obj', verbosity=0, subset_low=0, subset_high=None):
    assert isinstance(name, string_types)
    assert obj is not None
    assert type(subset_low) is int
    assert type(verbosity) is int

    if subset_high is None:
        subset = lambda ident: ident
    else:
        subset = partial(slice, subset_low, subset_high)

    print('type({}):'.format(name).ljust(16), '{}'.format(type(obj).__name__), sep='')
    if hasattr(obj, '__len__'):
        print('len({}):'.format(name).ljust(16), len(obj), sep='')
        # '\ndir({}):'.format(obj).ljust(16), dir(obj),

    if isinstance(obj, pd.Series):
        if verbosity > 0:
            print('{}.name:'.format(name).ljust(15), obj.name,
                  # '\tseries.columns:'.ljust(16), series.columns,
                  '\n{}.axes:'.format(name).ljust(16), obj.axes,
                  '\n{}.index:'.format(name).ljust(16), obj.index,
                  sep='')

        display(HTML(subset(obj.to_frame()).to_html()))
    elif isinstance(obj, pd.DataFrame):
        display(HTML(subset(obj).to_html()))
    else:
        print('{}:'.format(name).ljust(16), '{!r}'.format(obj), sep='')


def to_manycat_name(o):
    if isinstance(o, string_types):
        o = o,

    for e in o[::-1]:
        lower_e = e.lower()
        if lower_e == 'overall quality of the photographs taken':
            return 'Overall Quality of the Photographs Taken'
        elif e.startswith('ETDRS') or e == 'Overall Findings':
            return e
        elif 'macul' in lower_e:
            # print('matched with: {!r}'.format(e))
            return 'Maculopathy'
        elif e.startswith('Overall Finding'):
            return 'Overall Findings'
        else:
            print('no match found for: {!r}'.format(e))

    raise TypeError('{!r} no key found for'.format(o))


def app(idx, cat0):
    def inner(value):
        if app.t > 0:
            app.t -= 1
            debug(value, 'value', subset_high=5)
            print('----------------------------')
        return value

    return inner


app.t = 2


def process_within_series(cell):
    if process_within_series.t > 0:
        debug(cell, 'cell')

    return cell


process_within_series.t = 2


def prepare():
    df = pd.read_excel('/'.join(('file://localhost',
                                 path.expanduser('~').replace(path.sep, '/'),
                                 'OneDrive - The University of Sydney (Students)',
                                 'DR SPOC - Graders 1 and 2.xlsx')),
                       skiprows=1, header=[0, 1], index_col=[0])
    df = df.transform(grad_mac2)

    display(HTML('<h2>Columns</h2>'))

    display(HTML(
        '<ul>\n{}\n</ul>'.format('\n'.join(
            '  <li>"{}"</li>'.format(col)
            for col in chain_unique(map(itemgetter(1), df.axes[1]))
        ))))

    axes = filter(lambda c: c[:2] in image_angle_types,
                  map(itemgetter(0), df.axes[1]))
    columns = filterfalse(
        partial(contains,
                frozenset(
                    ('Overall quality of the photographs taken',
                     'Overall Finding'))),
        chain_unique(map(itemgetter(1), df.axes[1])))

    return df


def main():
    df = prepare()

    t = 4
    for i in df.index:
        # for col in columns:
        val = df.loc[i, :]  # type: pd.Series
        if t > 0:
            t -= 1
            debug(val, 'val')
        # val.apply(app(i, ))
        # if not np.isnan(val):
        val_df = val.to_frame()  # type: pd.DataFrame
        for tup in val_df.iteritems():
            if t > 0:
                print('-' * 67)

                for cell in tup:
                    debug(cell, 'cell')

                    if isinstance(cell, pd.Series):
                        cell = cell.apply(process_within_series)

                print('-' * 67, '\n')


if __name__ == '__main__':
    main()
