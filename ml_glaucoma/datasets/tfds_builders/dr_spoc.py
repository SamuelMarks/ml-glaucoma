#!/usr/bin/env python

import sys
from collections import Counter
from functools import partial
from itertools import filterfalse, chain
from operator import itemgetter, contains
from os import path, environ, listdir

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


def isnotebook():  # type: () -> bool
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
    def from_s(value, idx):  # type: (np.float, np.float) -> str or np.nan
        if pd.isnull(value) or isinstance(value, string_types):
            return value
        value = np.ushort(value)
        name = series.name if series.name in manycat2threecat else to_manycat_name(series.name)

        mapped = manycat2threecat.get(name)

        try:
            result = value if mapped is None or len(mapped) < value else mapped[4 if value == 5 else value]
        except IndexError:
            just = 20
            print('mapped:'.ljust(just), '{!r}\n'.format(mapped),
                  'value:'.ljust(just), '{!r}\n'.format(value),
                  'idx:'.ljust(just), '{!r}\n'.format(idx),
                  sep='')
            print(series.index)
            raise
        return result

    return series if series is None else series.apply(from_s, args=(series.name,))


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


def prepare(dr_spoc_dir, sheet_name):  # type: (str, str) -> pd.DataFrame
    df = pd \
        .read_excel('/'.join(('file://localhost',
                              path.dirname(path.dirname(dr_spoc_dir)).replace(path.sep, '/'),
                              'DR SPOC - Graders 1 and 2.xlsx')),
                    sheet_name=sheet_name,
                    skiprows=1,
                    header=[0, 1],
                    index_col=[0]) \
        .transform(grad_mac2)

    if prepare.t > 0:
        prepare.t -= 1
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


prepare.t = 0


def retrieve_from_db(dr_spoc_dir):  # type: () -> (pd.DataFrame, Counter)
    engine = create_engine(environ['RDBMS_URI'])

    with engine.connect() as con:
        r = con.execute(text('''
        CREATE OR REPLACE FUNCTION url_decode(input text) RETURNS text
            LANGUAGE plpgsql
            IMMUTABLE STRICT AS
        $$
        DECLARE
            bin  bytea = '';
            byte text;
        BEGIN
            FOR byte IN (select (regexp_matches(input, '(%..|.)', 'g'))[1])
                LOOP
                    IF length(byte) = 3 THEN
                        bin = bin || decode(substring(byte, 2, 2), 'hex');
                    ELSE
                        bin = bin || byte::bytea;
                    END IF;
                END LOOP;
            RETURN convert_from(bin, 'utf8');
        END
        $$;
        '''))

    assert r.rowcount == -1

    df = pd.read_sql('''
    SELECT replace(url_decode("artifactLocation"), 'fundus_images',
                   '{dr_spoc_dir_parent}') as "artifactLocation",
           category
    FROM categorise_tbl
    WHERE username = 'hamo.dw@gmail.com';
    '''.format(dr_spoc_dir_parent=path.dirname(dr_spoc_dir)), con=engine).set_index('artifactLocation')

    category2location = {cat: [] for cat in df.category.unique()}

    fname_co = Counter()

    def partition(series):  # type: (pd.Series) -> pd.Series
        def part(category, folder_name):  # type: (str, str) -> str
            if partition.t > 0:
                partition.t -= 1
                print('category:', category, '\n',
                      'folder_name:', folder_name, '\n', sep='\t')

            category2location[category].append(folder_name)
            fname_co[folder_name] += 1

            # symlink here?

            return category

        return series.apply(part, args=(series.name,))

    partition.t = 0

    df.apply(partition, 1)

    return df, fname_co


def construct_filename(dr_spoc_dir, image_position, folder_name):  # type: (str, str, int) -> str or np.nan
    if pd.isnull(image_position) or pd.isnull(folder_name):
        return np.nan
    image_position = image_position[:2]
    directory = path.join(dr_spoc_dir, 'DR SPOC Photo Dataset', str(folder_name))

    image = next((img
                  for img in listdir(directory)
                  if img.endswith('.jpg') and image_position in img),
                 np.nan)

    return image if pd.isnull(image) else path.join(directory, image)


def choice_between(record0, record1):  # type: (pd.Series, pd.Series) -> pd.Series
    if record0.choice == 'No gradable image':
        return record0
    elif record1.choice == 'No gradable image':
        return record1
    elif record0.choice == 'referable':
        return record0
    elif record1.choice == 'referable':
        return record1

    return record0


def find_compare(record, with_df):  # type: (pd.Series, pd.DataFrame) -> pd.Series
    try:
        record1 = with_df.loc[record.folder_name, (record.image_position, record.category)]
    except KeyError:
        return record

    return choice_between(record, record1)


def prepare_next(dr_spoc_dir):  # type: (str) -> (pd.DataFrame, pd.DataFrame)
    assert path.isdir(dr_spoc_dir)
    assert path.isdir(path.join(dr_spoc_dir, 'DR SPOC Photo Dataset'))
    assert not path.isdir(path.join(dr_spoc_dir, 'DR SPOC Photo Dataset', 'DR SPOC Dataset'))

    df_grader_1, df_grader_2 = (prepare(dr_spoc_dir, sheet_name='Grader {:d}'.format(i)) for i in (1, 2))
    just = 20

    # parseFname('DR SPOC Photo Dataset/6146/Upload/WA112325R2-8.jpg')

    filename_c = Counter()

    def fn(image_position, category, folder_name, choice):  # type: (str, str, int, str) -> pd.Series
        if not isinstance(choice, string_types):
            choice = np.nan

        if image_position == category:
            image_position = np.nan

        filename = construct_filename(dr_spoc_dir, image_position, folder_name)
        filename_c[filename] += 1

        series_input = {
            'image_position': image_position,
            'category': category,
            'folder_name': folder_name,
            'choice': choice
        }
        if fn.t > 0:
            fn.t -= 1
            print(
                'image_position:'.ljust(just), '{!r}'.format(image_position), '\n',
                'category:'.ljust(just), '{!r}'.format(category), '\n',
                'folder_name:'.ljust(just), '{!r}'.format(folder_name), '\n',
                'choice:'.ljust(just), '{!r}'.format(choice), '\n',
                'construct_filename:'.ljust(just), '{!r}'.format(filename), '\n',
                'pd.Series(series_input, index=sorted(series_input.keys())).index:',
                pd.Series(series_input, index=sorted(series_input.keys())).index, '\n',
                sep=''
            )

        return pd.Series(series_input, index=sorted(series_input.keys()))
        # return '_'.join(map(str, (image_position, category, folder_name, choice)))

    fn.t = 0

    return tuple(df.apply(lambda x: [fn(x.name[0], x.name[1], pos, value)
                                     for pos, value in x.items()])
                 for df in (df_grader_1, df_grader_2))


def main():  # type: () -> pd.DataFrame
    dr_spoc_dir = path.join(path.expanduser('~'),
                            'OneDrive - The University of Sydney (Students)',
                            'Fundus Photographs for AI',
                            'DR SPOC Dataset')
    df_grader_1, df_grader_2 = prepare_next(dr_spoc_dir=dr_spoc_dir)
    df = df_grader_1.transform(lambda series: pd.Series({k: find_compare(v, with_df=df_grader_2)
                                                         for k, v in series.items()}))
    frame_checks(dr_spoc_dir=dr_spoc_dir)
    return df


# :::::::::::::::::::::::::::::::::::::::

def frame_checks(dr_spoc_dir):
    db_df, db_fname_co = retrieve_from_db(dr_spoc_dir=dr_spoc_dir)

    # assert sum(map(itemgetter(1), filename_c.most_common()[1:])) // 3 == 1570
    assert sum(map(itemgetter(1), db_fname_co.most_common())) == 589

    print('db_fname_co:'.ljust(20), '{:04d}'.format(sum(map(itemgetter(1), db_fname_co.most_common()))))
    # print('filename_c:'.ljust(just), '{:04d}'.format(sum(map(itemgetter(1), filename_c.most_common()[1:])) // 3))


# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# print('filename_c.most_common()')
# print(filename_c.most_common())
# print('db_fname_co.most_common()')
# print(db_fname_co.most_common())
# total_filenames_c = filename_c + db_fname_co
# print(total_filenames_c.most_common())

# create new dataframe with
# [fname -> diagnoses]

# with open('/tmp/fnames.txt', 'wt') as f:
#    f.write('\n'.join(filter(None, map(
#        lambda c: c.replace('/Users/samuel/OneDrive - The University of Sydney (Students)/', '') if not pd.isnull(
#            c) else None, total_filenames_c.keys()))))


if __name__ == '__main__':
    main()
