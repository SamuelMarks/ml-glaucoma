ml_glaucoma
===========
A Glaucoma diagnosing CNN

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .

## CLI usage

    $ python -m ml_glaucoma --help

    usage: python -m ml_glaucoma [-h] [--version]
                             {download,vis,train,evaluate,parser} ...

    CLI for a Glaucoma diagnosing CNN
    
    positional arguments:
      {download,vis,train,evaluate,parser}
        download            Download and prepare required data
        vis                 Visualise data
        train               Train model
        evaluate            Evaluate model
        parser              Parse out metrics from log output. Default: per epoch
                            sensitivity & specificity.
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit

### `download`

    $ python -m ml_glaucoma download --help

    usage: python -m ml_glaucoma download [-h]
                                      [-ds {bmes,refuge} [{bmes,refuge} ...]]
                                      [--data_dir DATA_DIR]
                                      [--download_dir DOWNLOAD_DIR]
                                      [--extract_dir EXTRACT_DIR]
                                      [--manual_dir MANUAL_DIR]
                                      [--download_mode {reuse_dataset_if_exists,reuse_cache_if_exists,force_redownload}]
                                      [-r RESOLUTION RESOLUTION]
                                      [--gray_on_disk] [--bmes_init]
                                      [--bmes_parent_dir BMES_PARENT_DIR]

    optional arguments:
      -h, --help            show this help message and exit
      -ds {bmes,refuge} [{bmes,refuge} ...], --dataset {bmes,refuge} [{bmes,refuge} ...]
                            dataset key
      --data_dir DATA_DIR   root directory to store processed tfds records
      --download_dir DOWNLOAD_DIR
                            directory to store downloaded files
      --extract_dir EXTRACT_DIR
                            directory where extracted files are stored
      --manual_dir MANUAL_DIR
                            directory where manually downloaded files are saved
      --download_mode {reuse_dataset_if_exists,reuse_cache_if_exists,force_redownload}
                            tfds.GenerateMode
      -r RESOLUTION RESOLUTION, --resolution RESOLUTION RESOLUTION
                            image resolution
      --gray_on_disk        whether or not to save data as grayscale on disk
      --bmes_init           initial bmes get_data
      --bmes_parent_dir BMES_PARENT_DIR
                            parent directory of bmes data

### `vis`

    $ python -m ml_glaucoma vis --help
    usage: python -m ml_glaucoma vis [-h] [-ds {bmes,refuge} [{bmes,refuge} ...]]
                                 [--data_dir DATA_DIR]
                                 [--download_dir DOWNLOAD_DIR]
                                 [--extract_dir EXTRACT_DIR]
                                 [--manual_dir MANUAL_DIR]
                                 [--download_mode {reuse_dataset_if_exists,reuse_cache_if_exists,force_redownload}]
                                 [-r RESOLUTION RESOLUTION] [--gray_on_disk]
                                 [--bmes_init]
                                 [--bmes_parent_dir BMES_PARENT_DIR] [-fv]
                                 [-fh] [--gray]
                                 [-l {BinaryCrossentropy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,Hinge,Huber,KLD,KLDivergence,LogCosh,Loss,MAE,MAPE,MSE,MSLE,MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError,MeanSquaredLogarithmicError,Poisson,Reduction,SparseCategoricalCrossentropy,SquaredHinge,binary_crossentropy,categorical_crossentropy,categorical_hinge,cosine,cosine_proximity,cosine_similarity,deserialize,get,hinge,kld,kullback_leibler_divergence,logcosh,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,serialize,sparse_categorical_crossentropy,squared_hinge,BinaryCrossentropyWithRanking,DiceLoss,JaccardDistance,Kappa,PairLoss,SmoothL1,SoftAUC}]
                                 [-m [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} ...]]]
                                 [-pt [PRECISION_THRESHOLDS [PRECISION_THRESHOLDS ...]]]
                                 [-rt [RECALL_THRESHOLDS [RECALL_THRESHOLDS ...]]]
                                 [--shuffle_buffer SHUFFLE_BUFFER]
                                 [--use_inverse_freq_weights]

    optional arguments:
      -h, --help            show this help message and exit
      -ds {bmes,refuge} [{bmes,refuge} ...], --dataset {bmes,refuge} [{bmes,refuge} ...]
                            dataset key
      --data_dir DATA_DIR   root directory to store processed tfds records
      --download_dir DOWNLOAD_DIR
                            directory to store downloaded files
      --extract_dir EXTRACT_DIR
                            directory where extracted files are stored
      --manual_dir MANUAL_DIR
                            directory where manually downloaded files are saved
      --download_mode {reuse_dataset_if_exists,reuse_cache_if_exists,force_redownload}
                            tfds.GenerateMode
      -r RESOLUTION RESOLUTION, --resolution RESOLUTION RESOLUTION
                            image resolution
      --gray_on_disk        whether or not to save data as grayscale on disk
      --bmes_init           initial bmes get_data
      --bmes_parent_dir BMES_PARENT_DIR
                            parent directory of bmes data
      -fv, --maybe_vertical_flip
                            randomly flip training input vertically
      -fh, --maybe_horizontal_flip
                            randomly flip training input horizontally
      --gray                use grayscale
      -l {BinaryCrossentropy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,Hinge,Huber,KLD,KLDivergence,LogCosh,Loss,MAE,MAPE,MSE,MSLE,MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError,MeanSquaredLogarithmicError,Poisson,Reduction,SparseCategoricalCrossentropy,SquaredHinge,binary_crossentropy,categorical_crossentropy,categorical_hinge,cosine,cosine_proximity,cosine_similarity,deserialize,get,hinge,kld,kullback_leibler_divergence,logcosh,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,serialize,sparse_categorical_crossentropy,squared_hinge,BinaryCrossentropyWithRanking,DiceLoss,JaccardDistance,Kappa,PairLoss,SmoothL1,SoftAUC}, --loss {BinaryCrossentropy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,Hinge,Huber,KLD,KLDivergence,LogCosh,Loss,MAE,MAPE,MSE,MSLE,MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError,MeanSquaredLogarithmicError,Poisson,Reduction,SparseCategoricalCrossentropy,SquaredHinge,binary_crossentropy,categorical_crossentropy,categorical_hinge,cosine,cosine_proximity,cosine_similarity,deserialize,get,hinge,kld,kullback_leibler_divergence,logcosh,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,serialize,sparse_categorical_crossentropy,squared_hinge,BinaryCrossentropyWithRanking,DiceLoss,JaccardDistance,Kappa,PairLoss,SmoothL1,SoftAUC}
                            loss function to use
      -m [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} ...]], --metrics [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} ...]]
                            metric functions to use
      -pt [PRECISION_THRESHOLDS [PRECISION_THRESHOLDS ...]], --precision_thresholds [PRECISION_THRESHOLDS [PRECISION_THRESHOLDS ...]]
                            precision thresholds
      -rt [RECALL_THRESHOLDS [RECALL_THRESHOLDS ...]], --recall_thresholds [RECALL_THRESHOLDS [RECALL_THRESHOLDS ...]]
                            recall thresholds
      --shuffle_buffer SHUFFLE_BUFFER
                            buffer used in tf.data.Dataset.shuffle
      --use_inverse_freq_weights
                            weight loss according to inverse class frequency

### `train`

    $ python -m ml_glaucoma train --help

    usage: python -m ml_glaucoma train [-h]
                                       [-ds {bmes,refuge} [{bmes,refuge} ...]]
                                       [--data_dir DATA_DIR]
                                       [--download_dir DOWNLOAD_DIR]
                                       [--extract_dir EXTRACT_DIR]
                                       [--manual_dir MANUAL_DIR]
                                       [--download_mode {reuse_dataset_if_exists,reuse_cache_if_exists,force_redownload}]
                                       [-r RESOLUTION RESOLUTION] [--gray_on_disk]
                                       [--bmes_init]
                                       [--bmes_parent_dir BMES_PARENT_DIR] [-fv]
                                       [-fh] [--gray]
                                       [-l {BinaryCrossentropy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,Hinge,Huber,KLD,KLDivergence,LogCosh,Loss,MAE,MAPE,MSE,MSLE,MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError,MeanSquaredLogarithmicError,Poisson,Reduction,SparseCategoricalCrossentropy,SquaredHinge,binary_crossentropy,categorical_crossentropy,categorical_hinge,cosine,cosine_proximity,cosine_similarity,deserialize,get,hinge,kld,kullback_leibler_divergence,logcosh,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,serialize,sparse_categorical_crossentropy,squared_hinge,BinaryCrossentropyWithRanking,DiceLoss,JaccardDistance,Kappa,PairLoss,SmoothL1,SoftAUC}]
                                       [-m [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} ...]]]
                                       [-pt [PRECISION_THRESHOLDS [PRECISION_THRESHOLDS ...]]]
                                       [-rt [RECALL_THRESHOLDS [RECALL_THRESHOLDS ...]]]
                                       [--shuffle_buffer SHUFFLE_BUFFER]
                                       [--use_inverse_freq_weights]
                                       [--model_file [MODEL_FILE [MODEL_FILE ...]]]
                                       [--model_param [MODEL_PARAM [MODEL_PARAM ...]]]
                                       [-o {Adadelta,Adagrad,Adam,Adamax,Ftrl,Nadam,Optimizer,RMSprop,SGD}]
                                       [-lr LEARNING_RATE]
                                       [--exp_lr_decay EXP_LR_DECAY]
                                       [-b BATCH_SIZE] [-e EPOCHS]
                                       [--class-weight CLASS_WEIGHT]
                                       [--callback [{BaseLogger,CSVLogger,Callback,EarlyStopping,History,LambdaCallback,LearningRateScheduler,ModelCheckpoint,ProgbarLogger,ReduceLROnPlateau,RemoteMonitor,TensorBoard,TerminateOnNaN,AucRocCallback,ExponentialDecayLrSchedule,LoadingModelCheckpoint,SGDRScheduler} [{BaseLogger,CSVLogger,Callback,EarlyStopping,History,LambdaCallback,LearningRateScheduler,ModelCheckpoint,ProgbarLogger,ReduceLROnPlateau,RemoteMonitor,TensorBoard,TerminateOnNaN,AucRocCallback,ExponentialDecayLrSchedule,LoadingModelCheckpoint,SGDRScheduler} ...]]]
                                       [--model_dir MODEL_DIR]
                                       [-c CHECKPOINT_FREQ]
                                       [--summary_freq SUMMARY_FREQ]
                                       [-tb TB_LOG_DIR] [--write_images]
    
    optional arguments:
      -h, --help            show this help message and exit
      -ds {bmes,refuge} [{bmes,refuge} ...], --dataset {bmes,refuge} [{bmes,refuge} ...]
                            dataset key
      --data_dir DATA_DIR   root directory to store processed tfds records
      --download_dir DOWNLOAD_DIR
                            directory to store downloaded files
      --extract_dir EXTRACT_DIR
                            directory where extracted files are stored
      --manual_dir MANUAL_DIR
                            directory where manually downloaded files are saved
      --download_mode {reuse_dataset_if_exists,reuse_cache_if_exists,force_redownload}
                            tfds.GenerateMode
      -r RESOLUTION RESOLUTION, --resolution RESOLUTION RESOLUTION
                            image resolution
      --gray_on_disk        whether or not to save data as grayscale on disk
      --bmes_init           initial bmes get_data
      --bmes_parent_dir BMES_PARENT_DIR
                            parent directory of bmes data
      -fv, --maybe_vertical_flip
                            randomly flip training input vertically
      -fh, --maybe_horizontal_flip
                            randomly flip training input horizontally
      --gray                use grayscale
      -l {BinaryCrossentropy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,Hinge,Huber,KLD,KLDivergence,LogCosh,Loss,MAE,MAPE,MSE,MSLE,MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError,MeanSquaredLogarithmicError,Poisson,Reduction,SparseCategoricalCrossentropy,SquaredHinge,binary_crossentropy,categorical_crossentropy,categorical_hinge,cosine,cosine_proximity,cosine_similarity,deserialize,get,hinge,kld,kullback_leibler_divergence,logcosh,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,serialize,sparse_categorical_crossentropy,squared_hinge,BinaryCrossentropyWithRanking,DiceLoss,JaccardDistance,Kappa,PairLoss,SmoothL1,SoftAUC}, --loss {BinaryCrossentropy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,Hinge,Huber,KLD,KLDivergence,LogCosh,Loss,MAE,MAPE,MSE,MSLE,MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError,MeanSquaredLogarithmicError,Poisson,Reduction,SparseCategoricalCrossentropy,SquaredHinge,binary_crossentropy,categorical_crossentropy,categorical_hinge,cosine,cosine_proximity,cosine_similarity,deserialize,get,hinge,kld,kullback_leibler_divergence,logcosh,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,serialize,sparse_categorical_crossentropy,squared_hinge,BinaryCrossentropyWithRanking,DiceLoss,JaccardDistance,Kappa,PairLoss,SmoothL1,SoftAUC}
                            loss function to use
      -m [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} ...]], --metrics [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} ...]]
                            metric functions to use
      -pt [PRECISION_THRESHOLDS [PRECISION_THRESHOLDS ...]], --precision_thresholds [PRECISION_THRESHOLDS [PRECISION_THRESHOLDS ...]]
                            precision thresholds
      -rt [RECALL_THRESHOLDS [RECALL_THRESHOLDS ...]], --recall_thresholds [RECALL_THRESHOLDS [RECALL_THRESHOLDS ...]]
                            recall thresholds
      --shuffle_buffer SHUFFLE_BUFFER
                            buffer used in tf.data.Dataset.shuffle
      --use_inverse_freq_weights
                            weight loss according to inverse class frequency
      --model_file [MODEL_FILE [MODEL_FILE ...]]
                            gin files for model definition. Should define
                            `model_fn` macro either here or in --gin_param
      --model_param [MODEL_PARAM [MODEL_PARAM ...]]
                            gin_params for model definition. Should define
                            `model_fn` macro either here or in --gin_file
      -o {Adadelta,Adagrad,Adam,Adamax,Ftrl,Nadam,Optimizer,RMSprop,SGD}, --optimizer {Adadelta,Adagrad,Adam,Adamax,Ftrl,Nadam,Optimizer,RMSprop,SGD}
                            class name of optimizer to use
      -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                            base optimizer learning rate
      --exp_lr_decay EXP_LR_DECAY
                            exponential learning rate decay factor applied per
                            epoch, e.g. 0.98. None is interpreted as no decay
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            size of each batch
      -e EPOCHS, --epochs EPOCHS
                            number of epochs to run training from
      --class-weight CLASS_WEIGHT
                            Optional dictionary mapping class indices (integers)to
                            a weight (float) value, used for weighting the loss
                            function(during training only).This can be useful to
                            tell the model to"pay more attention" to samples
                            froman under-represented class.
      --callback [{BaseLogger,CSVLogger,Callback,EarlyStopping,History,LambdaCallback,LearningRateScheduler,ModelCheckpoint,ProgbarLogger,ReduceLROnPlateau,RemoteMonitor,TensorBoard,TerminateOnNaN,AucRocCallback,ExponentialDecayLrSchedule,LoadingModelCheckpoint,SGDRScheduler} [{BaseLogger,CSVLogger,Callback,EarlyStopping,History,LambdaCallback,LearningRateScheduler,ModelCheckpoint,ProgbarLogger,ReduceLROnPlateau,RemoteMonitor,TensorBoard,TerminateOnNaN,AucRocCallback,ExponentialDecayLrSchedule,LoadingModelCheckpoint,SGDRScheduler} ...]]
                            Keras callback function(s) to use. Extends default
                            callback list.
      --model_dir MODEL_DIR
                            model directory in which to save weights and
                            tensorboard summaries
      -c CHECKPOINT_FREQ, --checkpoint_freq CHECKPOINT_FREQ
                            epoch frequency at which to save model weights
      --summary_freq SUMMARY_FREQ
                            batch frequency at which to save tensorboard summaries
      -tb TB_LOG_DIR, --tb_log_dir TB_LOG_DIR
                            tensorboard_log_dir (defaults to model_dir)
      --write_images        whether or not to write images to tensorboard

### `evaluate`

    $ python -m ml_glaucoma evaluate --help

    usage: python -m ml_glaucoma evaluate [-h]
                                          [-ds {bmes,refuge} [{bmes,refuge} ...]]
                                          [--data_dir DATA_DIR]
                                          [--download_dir DOWNLOAD_DIR]
                                          [--extract_dir EXTRACT_DIR]
                                          [--manual_dir MANUAL_DIR]
                                          [--download_mode {reuse_dataset_if_exists,reuse_cache_if_exists,force_redownload}]
                                          [-r RESOLUTION RESOLUTION]
                                          [--gray_on_disk] [--bmes_init]
                                          [--bmes_parent_dir BMES_PARENT_DIR]
                                          [-fv] [-fh] [--gray]
                                          [-l {BinaryCrossentropy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,Hinge,Huber,KLD,KLDivergence,LogCosh,Loss,MAE,MAPE,MSE,MSLE,MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError,MeanSquaredLogarithmicError,Poisson,Reduction,SparseCategoricalCrossentropy,SquaredHinge,binary_crossentropy,categorical_crossentropy,categorical_hinge,cosine,cosine_proximity,cosine_similarity,deserialize,get,hinge,kld,kullback_leibler_divergence,logcosh,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,serialize,sparse_categorical_crossentropy,squared_hinge,BinaryCrossentropyWithRanking,DiceLoss,JaccardDistance,Kappa,PairLoss,SmoothL1,SoftAUC}]
                                          [-m [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} ...]]]
                                          [-pt [PRECISION_THRESHOLDS [PRECISION_THRESHOLDS ...]]]
                                          [-rt [RECALL_THRESHOLDS [RECALL_THRESHOLDS ...]]]
                                          [--shuffle_buffer SHUFFLE_BUFFER]
                                          [--use_inverse_freq_weights]
                                          [--model_file [MODEL_FILE [MODEL_FILE ...]]]
                                          [--model_param [MODEL_PARAM [MODEL_PARAM ...]]]
                                          [-o {Adadelta,Adagrad,Adam,Adamax,Ftrl,Nadam,Optimizer,RMSprop,SGD}]
                                          [-lr LEARNING_RATE]
                                          [--exp_lr_decay EXP_LR_DECAY]
                                          [-b BATCH_SIZE] [-e EPOCHS]
                                          [--class-weight CLASS_WEIGHT]
                                          [--callback [{BaseLogger,CSVLogger,Callback,EarlyStopping,History,LambdaCallback,LearningRateScheduler,ModelCheckpoint,ProgbarLogger,ReduceLROnPlateau,RemoteMonitor,TensorBoard,TerminateOnNaN,AucRocCallback,ExponentialDecayLrSchedule,LoadingModelCheckpoint,SGDRScheduler} [{BaseLogger,CSVLogger,Callback,EarlyStopping,History,LambdaCallback,LearningRateScheduler,ModelCheckpoint,ProgbarLogger,ReduceLROnPlateau,RemoteMonitor,TensorBoard,TerminateOnNaN,AucRocCallback,ExponentialDecayLrSchedule,LoadingModelCheckpoint,SGDRScheduler} ...]]]
                                          [--model_dir MODEL_DIR]
                                          [-c CHECKPOINT_FREQ]
                                          [--summary_freq SUMMARY_FREQ]
                                          [-tb TB_LOG_DIR] [--write_images]
    
    optional arguments:
      -h, --help            show this help message and exit
      -ds {bmes,refuge} [{bmes,refuge} ...], --dataset {bmes,refuge} [{bmes,refuge} ...]
                            dataset key
      --data_dir DATA_DIR   root directory to store processed tfds records
      --download_dir DOWNLOAD_DIR
                            directory to store downloaded files
      --extract_dir EXTRACT_DIR
                            directory where extracted files are stored
      --manual_dir MANUAL_DIR
                            directory where manually downloaded files are saved
      --download_mode {reuse_dataset_if_exists,reuse_cache_if_exists,force_redownload}
                            tfds.GenerateMode
      -r RESOLUTION RESOLUTION, --resolution RESOLUTION RESOLUTION
                            image resolution
      --gray_on_disk        whether or not to save data as grayscale on disk
      --bmes_init           initial bmes get_data
      --bmes_parent_dir BMES_PARENT_DIR
                            parent directory of bmes data
      -fv, --maybe_vertical_flip
                            randomly flip training input vertically
      -fh, --maybe_horizontal_flip
                            randomly flip training input horizontally
      --gray                use grayscale
      -l {BinaryCrossentropy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,Hinge,Huber,KLD,KLDivergence,LogCosh,Loss,MAE,MAPE,MSE,MSLE,MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError,MeanSquaredLogarithmicError,Poisson,Reduction,SparseCategoricalCrossentropy,SquaredHinge,binary_crossentropy,categorical_crossentropy,categorical_hinge,cosine,cosine_proximity,cosine_similarity,deserialize,get,hinge,kld,kullback_leibler_divergence,logcosh,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,serialize,sparse_categorical_crossentropy,squared_hinge,BinaryCrossentropyWithRanking,DiceLoss,JaccardDistance,Kappa,PairLoss,SmoothL1,SoftAUC}, --loss {BinaryCrossentropy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,Hinge,Huber,KLD,KLDivergence,LogCosh,Loss,MAE,MAPE,MSE,MSLE,MeanAbsoluteError,MeanAbsolutePercentageError,MeanSquaredError,MeanSquaredLogarithmicError,Poisson,Reduction,SparseCategoricalCrossentropy,SquaredHinge,binary_crossentropy,categorical_crossentropy,categorical_hinge,cosine,cosine_proximity,cosine_similarity,deserialize,get,hinge,kld,kullback_leibler_divergence,logcosh,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,serialize,sparse_categorical_crossentropy,squared_hinge,BinaryCrossentropyWithRanking,DiceLoss,JaccardDistance,Kappa,PairLoss,SmoothL1,SoftAUC}
                            loss function to use
      -m [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} ...]], --metrics [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} [{AUC,Accuracy,BinaryAccuracy,BinaryCrossentropy,CategoricalAccuracy,CategoricalCrossentropy,CategoricalHinge,CosineSimilarity,FalseNegatives,FalsePositives,Hinge,KLD,KLDivergence,LogCoshError,MAE,MAPE,MSE,MSLE,Mean,MeanAbsoluteError,MeanAbsolutePercentageError,MeanIoU,MeanRelativeError,MeanSquaredError,MeanSquaredLogarithmicError,MeanTensor,Metric,Poisson,Precision,Recall,RootMeanSquaredError,SensitivityAtSpecificity,SparseCategoricalAccuracy,SparseCategoricalCrossentropy,SparseTopKCategoricalAccuracy,SpecificityAtSensitivity,SquaredHinge,Sum,TopKCategoricalAccuracy,TrueNegatives,TruePositives,binary_accuracy,binary_crossentropy,categorical_accuracy,categorical_crossentropy,cosine,cosine_proximity,hinge,kld,kullback_leibler_divergence,mae,mape,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,mean_squared_logarithmic_error,mse,msle,poisson,sparse_categorical_accuracy,sparse_categorical_crossentropy,sparse_top_k_categorical_accuracy,squared_hinge,top_k_categorical_accuracy,F1Metric} ...]]
                            metric functions to use
      -pt [PRECISION_THRESHOLDS [PRECISION_THRESHOLDS ...]], --precision_thresholds [PRECISION_THRESHOLDS [PRECISION_THRESHOLDS ...]]
                            precision thresholds
      -rt [RECALL_THRESHOLDS [RECALL_THRESHOLDS ...]], --recall_thresholds [RECALL_THRESHOLDS [RECALL_THRESHOLDS ...]]
                            recall thresholds
      --shuffle_buffer SHUFFLE_BUFFER
                            buffer used in tf.data.Dataset.shuffle
      --use_inverse_freq_weights
                            weight loss according to inverse class frequency
      --model_file [MODEL_FILE [MODEL_FILE ...]]
                            gin files for model definition. Should define
                            `model_fn` macro either here or in --gin_param
      --model_param [MODEL_PARAM [MODEL_PARAM ...]]
                            gin_params for model definition. Should define
                            `model_fn` macro either here or in --gin_file
      -o {Adadelta,Adagrad,Adam,Adamax,Ftrl,Nadam,Optimizer,RMSprop,SGD}, --optimizer {Adadelta,Adagrad,Adam,Adamax,Ftrl,Nadam,Optimizer,RMSprop,SGD}
                            class name of optimizer to use
      -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                            base optimizer learning rate
      --exp_lr_decay EXP_LR_DECAY
                            exponential learning rate decay factor applied per
                            epoch, e.g. 0.98. None is interpreted as no decay
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            size of each batch
      -e EPOCHS, --epochs EPOCHS
                            number of epochs to run training from
      --class-weight CLASS_WEIGHT
                            Optional dictionary mapping class indices (integers)to
                            a weight (float) value, used for weighting the loss
                            function(during training only).This can be useful to
                            tell the model to"pay more attention" to samples
                            froman under-represented class.
      --callback [{BaseLogger,CSVLogger,Callback,EarlyStopping,History,LambdaCallback,LearningRateScheduler,ModelCheckpoint,ProgbarLogger,ReduceLROnPlateau,RemoteMonitor,TensorBoard,TerminateOnNaN,AucRocCallback,ExponentialDecayLrSchedule,LoadingModelCheckpoint,SGDRScheduler} [{BaseLogger,CSVLogger,Callback,EarlyStopping,History,LambdaCallback,LearningRateScheduler,ModelCheckpoint,ProgbarLogger,ReduceLROnPlateau,RemoteMonitor,TensorBoard,TerminateOnNaN,AucRocCallback,ExponentialDecayLrSchedule,LoadingModelCheckpoint,SGDRScheduler} ...]]
                            Keras callback function(s) to use. Extends default
                            callback list.
      --model_dir MODEL_DIR
                            model directory in which to save weights and
                            tensorboard summaries
      -c CHECKPOINT_FREQ, --checkpoint_freq CHECKPOINT_FREQ
                            epoch frequency at which to save model weights
      --summary_freq SUMMARY_FREQ
                            batch frequency at which to save tensorboard summaries
      -tb TB_LOG_DIR, --tb_log_dir TB_LOG_DIR
                            tensorboard_log_dir (defaults to model_dir)
      --write_images        whether or not to write images to tensorboard

### `parser`
You can pipe or include a filename.

    $ python -m ml_glaucoma parser --help

    usage: python -m ml_glaucoma parser [-h] [--threshold THRESHOLD] [--top TOP]
                                        [--by-diff]
                                        [infile]
    
    Show metrics from output. Default: per epoch sensitivity & specificity.
    
    positional arguments:
      infile                File to work from. Defaults to stdin. So can pipe.
    
    optional arguments:
      -h, --help            show this help message and exit
      --threshold THRESHOLD
                            E.g.: 0.7 for sensitivity & specificity >= 70%
      --top TOP             Show top k results
      --by-diff             Sort by lowest difference between sensitivity &
                            specificity

# Project Structure

Training/validation scripts are provided in `data_preparation_scripts` and each call a function defined in `ml_glaucoma.runners`. We aim to provide highly-configurable runs, but the main parts to consider are:

* `problem`: the dataset, loss and metrics used during training
* `model_fn`: the function that takes one or more `tf.keras.layers.Input`s and returns a learnable keras model.

`model_fn`s are configured using using a forked [TF2.0 compatible gin-config](https://github.com/jackd/gin-config/tree/tf2) (awaiting on [this PR](https://github.com/google/gin-config/pull/17) before reverting to the [google version](https://github.com/google/gin-config.git). See example configs in `model_configs` and the [gin user guide](https://github.com/google/gin-config/blob/master/docs/index.md).

## Example usage:

```bash
python -m ml_glaucoma vis --dataset=refuge
python -m ml_glaucoma train \
  --model_file 'model_configs/dc.gin'  \
  --model_param 'import ml_glaucoma.gin_keras' 'dc0.kernel_regularizer=@tf.keras.regularizers.l2()' 'tf.keras.regularizers.l2.l = 1e-2' \
  --model_dir /tmp/ml_glaucoma/dc0-reg \
  -m BinaryAccuracy AUC \
  -pt 0.1 0.2 0.5 -rt 0.1 0.2 0.5 \
  --use_inverse_freq_weights
# ...
tensorboard --logdir=/tmp/ml_glaucoma
```

## Tensorflow Datasets

The main `Problem` implementation is backed by [tensorflow_datasets](https://github.com/tensorflow/datasets). This should manage dataset downloads, extraction, sha256 checks, on-disk shuffling/sharding and other best practices. Consequently it takes slightly longer to process initially, but the benefits in the long run are worth it.

## BMES Initialization

The current implementation leverages the existing `ml_glaucoma.utils.bmes_data_prep.get_data` method to separate files. This uses `tf.contrib` so requires `tf < 2.0`. It can be run using the `--bmes_init` flag within `python -m ml_glaucoma download`. This must be run prior to the standard `tfds.DatasetBuilder.download_and_prepare` which is run automatically if necessary. Once the `tfds` files have been generated, the original `get_data` directories are no longer required.

If the test/train/validation split here is just a random split, this could be done more easily by creating a single `tfds` split and using `tfds.Split.subsplit` - see [this post](https://www.tensorflow.org/datasets/splits).

## Status

* Automatic model saving/loading via modified `ModelCheckpoint`.
* Automatic tensorboard updates (fairly hacky interoperability with `ModelCheckpoint` to ensure restarted training runs have the appropriate step count).
* Loss re-weighting according to inverse class frequency (`TfdsProblem.use_inverse_freq_weights`).
* Only `dc0`, `applications` ([Keras applications](https://keras.io/applications)), `efficientnet` and `squeeze_excite_resnet` model verified to work. `dr0`, `dc1`, `dc2`, `dc3` and other squeeze excite networks implemented but untested.
* Only `refuge` and `bmes` dataset implemented, and only tested the classification task.
* BMES dataset: currently requires 2-stage preparation: `bmes_init` which is based on `ml_glaucoma.utils.bmes_data_prep.get_data` and the standard `tfds.DatasetBuilder.download_and_prepare`. The first stage will only be run if `--bmes_init` is used in `python -m ml_glaucoma download` arguments.
