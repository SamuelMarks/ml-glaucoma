from os import path, makedirs

from ml_glaucoma import get_logger

logger = get_logger(__file__.partition('.')[0])


def download(download_dir, force_new=False):
    from urllib3 import PoolManager
    import certifi

    http = PoolManager(cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())

    if not path.exists(download_dir):
        makedirs(download_dir)

    base = 'https://github.com/fchollet/deep-learning-models/releases/download'
    paths = '/v0.1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',

    # TODO: Concurrency
    for fname in paths:
        basename = fname[fname.rfind('/') + 1:]
        to_file = path.join(download_dir, basename)

        if not force_new and path.isfile(to_file):
            logger.info('File exists: {to_file}'.format(to_file=to_file))
            continue

        logger.info('Downloading: "{basename}" to: "{download_dir}"'.format(
            basename=basename, download_dir=download_dir
        ))

        r = http.request('GET', '{base}{fname}'.format(base=base, fname=fname),
                         preload_content=False)
        with open(to_file, 'wb') as f:
            for chunk in r.stream(32):
                f.write(chunk)
