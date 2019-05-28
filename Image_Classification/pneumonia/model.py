from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout,GlobalAveragePooling2D
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras import Input

class InceptionResnetV2Model():

    def __init__(self,shape,optimizer,activation,classes,loss):
        self.shape = shape
        self.optimizer = optimizer
        self.activation = activation
        self.classes = classes
        self.loss = loss

    def modelCreate(self):
        '''
        :return model:building model with InceptionResNetV2
        '''
        baseModel = InceptionResNetV2(include_top=False)
        inputs = Input(shape=self.shape)
        layer1 = baseModel(inputs)
        layer2 = GlobalAveragePooling2D()(layer1)
        dropout1 = Dropout(0.25)(layer2)
        outputs = Dense(self.classes, activation=self.activation)(dropout1)
        model = Model(inputs, outputs)
        model.compile( loss=self.loss,
                       optimizer=self.optimizer,
                       metrics=['accuracy']      )
        model.summary()

        return model

