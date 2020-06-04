from os import environ

if environ['TF']:
    from ml_glaucoma.cli_options.reprocess_datasets.tf import reprocess_datasets
elif environ['TORCH']:
    raise NotImplementedError()
else:
    raise NotImplementedError()
