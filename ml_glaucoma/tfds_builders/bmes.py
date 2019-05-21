from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
from ml_glaucoma.tfds_builders import transformer


def BmesConfig(resolution=None):
    return transformer.ImageTransformerConfig(
        description="TODO", resolution=resolution)


def _load_image(image_fp):
    return np.array(tfds.core.lazy_imports.PIL_Image.open(image_fp))


base_config = BmesConfig(None)


class Bmes(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [base_config]

    def _info(self):
        resolution = self.builder_config.resolution
        if resolution is None:
            h, w = None, None
        else:
            h, w = resolution
        return tfds.core.DatasetInfo(
            builder=self,
            description=self.builder_config.description,
            features=tfds.features.FeaturesDict({
                "fundus": tfds.features.Image(shape=(h, w, 3)),
                "label": tfds.features.Tensor(dtype=tf.bool, shape=()),
                "filename": tfds.features.Text(),
            }),
            urls=["https://github.com/SamuelMarks/ml-glaucoma"],
            citation="TODO",
            supervised_keys=("fundus", "label")
        )

    def _split_generators(self, dl_manager):
        generators = []
        manual_dir = dl_manager.manual_dir
        for split in ('train', 'validation', 'test'):
            folder = os.path.join(manual_dir, split)
            if not os.path.isdir(folder):
                raise IOError(
                    'No manually downloaded data found at %s' % folder)
            # assumes directory structure (filenames may vary)
            # manual dir
            # - test
            #  - no_glaucoma
            #   - neg_example1.jpg
            #   - neg_example2.jpg
            #  - glaucoma
            #   - pos_example1.jpg
            # - train
            #   - no_glaucoma
            #   - ...
            # - validation
            #   - no_glaucoma
            #   - ...
            num_examples = sum(
                len(os.listdir(os.path.join(folder, dirname)))
                for dirname in os.listdir(folder))
            num_shards = num_examples // 100 + 2  # basic heuristic
            generators.append(tfds.core.SplitGenerator(
                name=split,
                num_shards=num_shards,
                gen_kwargs=dict(folder=folder)))

        return generators

    def _generate_examples(self, folder):
        for dirname, label in (('no_glaucoma', False), ('glaucoma', True)):
            subdir = os.path.join(folder, subdir)
            for filename in os.listdir(subdir):
                if not filename.endswith('.jpg'):
                    raise IOError('All files in directory must be `.jpg`')
                path = os.path.join(subdir, filename)
                with tf.io.gfile.GFile(path, "rb") as fp:
                    fundus = _load_image(path)
                transformer = self.builder_config.transformer(fundus.shape[:2])
                if transformer is not None:
                    fundus = transformer.transform_image(
                        fundus, interp=tf.image.ResizeMethod.BILINEAR)
                yield dict(
                    fundus=fundus,
                    label=label,
                    filename=filename,
                )
