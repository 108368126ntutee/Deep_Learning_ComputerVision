import tensorflow as tf
import os, pathlib, random
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def getLabel(path):
    label_names = sorted( item.name for item in path.glob('*/') if item.is_dir() )
    label_to_index = dict( (name, index) for index,name in enumerate(label_names) )
    return label_to_index

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def writeLabelAsTFRecord(filename,labelData):
    writer = tf.io.TFRecordWriter(filename)
    for label in labelData:
        example = tf.train.Example( features=tf.train.Features(
                                         feature={'label': _int64_feature([label])})
                                  )

        writer.write(example.SerializeToString())
    writer.close()


baseDir = "/home/danny/Python_project/chest-xray-pneumonia/chest_xray/chest_xray"
trainDir = pathlib.Path(os.path.join(baseDir, "train1"))
validDir = pathlib.Path(os.path.join(baseDir, "valid1"))

trainImagePath = list(trainDir.glob('*/*'))
trainImagePath = [str(path) for path in trainImagePath]
random.shuffle(trainImagePath)

validImagePath = list(validDir.glob('*/*'))
validImagePath = [str(path) for path in validImagePath]
random.shuffle(validImagePath)

labelToIndex = getLabel(trainDir)
trainImageLabels = np.array([ labelToIndex[pathlib.Path(path).parent.name] for path in trainImagePath ])
validImageLabels = np.array([ labelToIndex[pathlib.Path(path).parent.name] for path in validImagePath ])