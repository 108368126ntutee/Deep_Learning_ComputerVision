import tensorflow as tf
import os
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, Flatten
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocessImage(image):
  image = tf.image.decode_jpeg(image,channels=3)
  image = tf.image.resize( image,[299,299] )
  image /= 255.0
  return image

@tf.function
def trainStep(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  trainLoss(loss)
  trainAccuracy(labels, predictions)

@tf.function
def validStep(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  validLoss(t_loss)
  validAccuracy(labels, predictions)

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(1, activation='sigmoid')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

class testModel(Model):
  def __init__(self):
    super(testModel, self).__init__()
    self.incpetionResNetBaseModel = InceptionResNetV2(include_top=False)
    self.GlobalAveragePool = GlobalAveragePooling2D()
    self.d1 = Dense(1, activation='sigmoid')

  def call(self, inputs):
    x = self.incpetionResNetBaseModel(inputs)
    x = self.GlobalAveragePool(x)
    return self.d1(x)

def _parse_function(example_proto):
    image_feature_description = {'label': tf.io.FixedLenFeature([], tf.int64)}
    example = tf.io.parse_single_example(example_proto, image_feature_description)
    example['label'] = example['label'][...,tf.newaxis]
    return example['label']

model = testModel()
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCHSIZE = 50
learningRate = 3e-5
EPOCHS = 3



trainImage = tf.data.TFRecordDataset("trainImage.tfrec").map(preprocessImage)
trainLabel = tf.data.TFRecordDataset("trainLabel.tfrec").map(_parse_function)

validImage = tf.data.TFRecordDataset("validImage.tfrec").map(preprocessImage)
validLabel = tf.data.TFRecordDataset("validLabel.tfrec").map(_parse_function)


trainDataset = tf.data.Dataset.zip((trainImage,trainLabel))
trainDataset = trainDataset.shuffle(buffer_size=3512)
trainDataset = trainDataset.batch(BATCHSIZE)

validDataset = tf.data.Dataset.zip((validImage,validLabel))
validDataset = validDataset.batch(BATCHSIZE)

loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learningRate)

trainLoss = tf.keras.metrics.Mean(name='train_loss')
trainAccuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

validLoss = tf.keras.metrics.Mean(name='valid_loss')
validAccuracy = tf.keras.metrics.BinaryAccuracy(name='valid_accuracy')


for epoch in range(EPOCHS):
    start = time.time()
    for images, labels in trainDataset:
        trainStep(images, labels)

    for valid_images, valid_labels in validDataset:
        validStep(valid_images, valid_labels)
    end = time.time()
    outputPrint = f"EPoch: {epoch+1} , Running time : {end - start:.2f} sec , Loss: {trainLoss.result():.3f} , " \
                  f" Accuracy: {trainAccuracy.result()*100:.2f} % " \
                  f", valid_loss: {validLoss.result():.3f} , valid Accuracy:{validAccuracy.result()*100:.2f} % "

    print(outputPrint)





