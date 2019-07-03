from ast import parse
from distutils.sysconfig import get_python_lib
from functools import partial
from os import path, listdir
from platform import python_version_tuple

from setuptools import setup, find_packages

if python_version_tuple()[0] == '2':
    from itertools import imap as map, ifilter as filter

if __name__ == '__main__':
    package_name = 'ml_glaucoma'

    with open(path.join(package_name, '__init__.py')) as f:
        __author__, __version__ = map(
            lambda buf: next(map(lambda e: e.value.s, parse(buf).body)),
            filter(lambda line: line.startswith('__version__') or line.startswith('__author__'), f)
        )

    to_funcs = lambda *paths: (partial(path.join, path.dirname(__file__), package_name, *paths),
                               partial(path.join, get_python_lib(prefix=''), package_name, *paths))

    _data_join, _data_install_dir = to_funcs('_data')
    _data_cache_join, _data_cache_install_dir = to_funcs('_data', '.cache')
    _url_checksums_join, _url_checksums_install_dir = to_funcs('url_checksums')
    _model_configs_join, _model_configs_install_dir = to_funcs('model_configs')

    setup(
        name=package_name,
        author=__author__,
        version=__version__,
        test_suite=package_name + '.tests',
        packages=find_packages(),
        package_dir={package_name: package_name},
        data_files=[
            (_data_install_dir(), filter(lambda p: path.isfile(p), list(map(_data_join, listdir(_data_join()))))),
            (_data_cache_install_dir(), list(map(_data_cache_join, listdir(_data_cache_join())))),
            (_url_checksums_install_dir(), list(map(_url_checksums_join, listdir(_url_checksums_join())))),
            (_model_configs_install_dir(), list(map(_model_configs_join, listdir(_model_configs_join()))))
        ]
    )
