
class trainModel():

    def __init__(self,epochs,validationSteps,stepsPerEpoch,trainDataGen,validDataGen):
        self.epochs = epochs
        self.validationSteps = validationSteps
        self.stepsPerEpoch = stepsPerEpoch
        self.trainDataGen = trainDataGen
        self.validDataGen = validDataGen

    def startTrain(self,model):

        trainHistory = model.fit_generator( self.trainDataGen,
                                            validation_data=self.validDataGen,
                                            validation_steps=self.validationSteps,
                                            steps_per_epoch=self.stepsPerEpoch,
                                            epochs=self.epochs                     )
        model.save("test0512.h5")
        return trainHistory

