from tensorflow.keras.preprocessing.image import ImageDataGenerator

class loadImgGenFromDir():

    def __init__(self,targetDir,shape,classMode,batchSize=32,rotateAngle=None,widthShift=None,heightShift=None,horizon=False,vertical=False):
        self.targetDir = targetDir
        self.shape = shape
        self.classMode = classMode
        self.batchSize = batchSize
        self.rotateAngle = rotateAngle
        self.widthShift = widthShift
        self.heigthShift = heightShift
        self.horizon = horizon
        self.vertical = vertical

    def loadByGenFlowFromDir(self):

        imgGen = ImageDataGenerator(rescale=1. / 255 ,
                                     rotation_range=self.rotateAngle,
                                     width_shift_range=self.widthShift,
                                     height_shift_range=self.heigthShift,
                                     horizontal_flip=self.horizon,
                                     vertical_flip=self.vertical          )

        imgGenFlowFromDir = imgGen.flow_from_directory(     self.targetDir,
                                              target_size=self.shape,
                                              class_mode=self.classMode,
                                              batch_size=self.batchSize             )
        return imgGenFlowFromDir