import pandas as pd

    import numpy as np

    #loading the dataset
    #df = pd.read_csv('/content/Tensorflow_object_detector/workspace/training_OD/annotations/dataset_out.csv',error_bad_lines=False)
    df = pd.read_csv('/content/Tensorflow_object_detector/workspace/training_OD/annotations/dataset_out2.csv',error_bad_lines=False)
    df['split'] = np.random.randn(df.shape[0], 1)

    msk = np.random.rand(len(df)) <= 0.8

    train = df[msk]

    test = df[~msk]

    train.to_csv('/content/Tensorflow_object_detector/workspace/training_OD/annotations/train.csv', index=None)
    test.to_csv('/content/Tensorflow_object_detector/workspace/training_OD/annotations/test.csv', index=None)