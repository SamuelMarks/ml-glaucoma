#!/usr/bin/env python
import sys
from collections import Counter
from functools import partial
from itertools import filterfalse, chain, repeat
from operator import itemgetter, contains
from os import path, environ

import numpy as np
import pandas as pd

from ml_glaucoma.utils import pp


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
elif environ.get('OUTPUT_HTML'):
    display = print
    HTML = lambda ident: ident


    def new_p(*args, sep=''):
        s, prev = '<p>', ''
        for arg in args:
            s += '{}{arg}{sep}'.format('&#9;' if isinstance(prev, string_types) and prev.rstrip().endswith(':')
                                       else '</p><p>',
                                       arg=arg, sep=sep)
            prev = arg
        s += '</p>'
        sys.stdout.write(s)
        sys.stdout.flush()


    print = new_p
else:
    HTML = lambda ident: ident
    display = print

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


def to_manycat_name(o):  # type: ([str]) -> str
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


def grad_mac2(series):  # type: (pd.Series) -> pd.Series
    def from_s(value):  # type: (np.float) -> str or np.nan
        if pd.isnull(value) or isinstance(value, string_types):
            return value
        value = np.ushort(value)
        name = series.name if series.name in manycat2threecat else to_manycat_name(series.name)

        mapped = manycat2threecat.get(name)

        return value if mapped is None or len(mapped) < value else mapped[value]

    return series if series is None else series.apply(from_s)


def debug(obj, name='obj', verbosity=0, subset_low=0, subset_high=None):  # type: (any, str, int, int, int) -> None
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


def to_manycat_name(o):  # type: ([str]) -> str
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


def prepare():  # type: () -> pd.DataFrame
    df = pd \
        .read_excel('/'.join(('file://localhost',
                              path.expanduser('~').replace(path.sep, '/'),
                              'OneDrive - The University of Sydney (Students)',
                              'DR SPOC - Graders 1 and 2.xlsx')),
                    skiprows=1, header=[0, 1], index_col=[0]) \
        .transform(grad_mac2)

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


def main():  # type: () -> None
    df = prepare()
    just = 20
    # image_position_c, category_c, folder_name_c, choice_c = repeat(Counter(), 4)

    image_position_c = Counter()
    category_c = Counter()
    folder_name_c = Counter()
    choice_c = Counter()

    def fn(image_position, category, folder_name, choice):  # type: (str, str, int, str) -> str
        def maybe_nan_str(o):
            return 'NaN' if pd.isnull(o) else o

        if not isinstance(image_position, string_types):
            image_position = np.nan
        if not isinstance(category, string_types):
            category = np.nan
        if type(folder_name) is not int:
            folder_name = np.nan
        if not isinstance(choice, string_types):
            choice = np.nan

        if image_position == category:
            image_position = np.nan

        image_position_c[maybe_nan_str(image_position)] += 1
        category_c[maybe_nan_str(category)] += 1
        folder_name_c[maybe_nan_str(folder_name)] += 1
        choice_c[maybe_nan_str(choice)] += 1
        if fn.t > 0:
            fn.t -= 1
            print('image_position:'.ljust(just), '{!r}'.format(image_position), '\n',
                  'category:'.ljust(just), '{!r}'.format(category), '\n',
                  'folder_name:'.ljust(just), '{!r}'.format(folder_name), '\n',
                  'choice:'.ljust(just), '{!r}'.format(choice), '\n',
                  sep='')
        return '_'.join(map(str, (image_position, category, folder_name, choice)))

    fn.t = 0

    df.transform(lambda x: [fn(x.name[0], x.name[1], pos, value)
                            for pos, value in x.items()])

    # engine = create_engine(environ['RDBMS_URI'])


'''
image_position:     R1 (Right macula-centred image)
category:           Overall quality of the photographs taken
folder_name:        27
---------------------

image_position:     R1 (Right macula-centred image)
category:           ETDRS Grading
folder_name:        27
---------------------

image_position:     R1 (Right macula-centred image)
category:           Maculopathy
folder_name:        27
---------------------
'''

if __name__ == '__main__':
    main()
