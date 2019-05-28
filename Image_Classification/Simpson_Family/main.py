from model import InceptionResnetV2Model
from loadData import loadImgGenFromDir
from train import trainModel
from tensorflow.keras import optimizers
import os
import matplotlib.pyplot as plt


shape = (299,299,3)
learningRate = 3e-5
optimizer = optimizers.Adam(learningRate)
activation = "softmax"
classes = 26
loss = "categorical_crossentropy"
baseDir = '/home/danny/Python_project/simpson/the-simpsons-characters-dataset'
trainDir = os.path.join( baseDir , 'train' )
validDir = os.path.join( baseDir , 'valid' )
batchSize = 20
epochs = 70
validationSteps = int(2029/batchSize)+1
stepsPerEpoch = int(16215 /batchSize)+1

def showTrainHistory(trainingHistory):

    plt.figure(figsize=(20, 10))
    plt.suptitle('Train History', fontsize=30)

    title = ["accuracy", "loss"]
    trainOutcome = {"accuracy": ["acc", "val_acc"], "loss": ["loss", "val_loss"]}

    for position in range(2):
        plt.subplot(1, 2, position + 1)
        plt.title(title[position])
        for outcome in range(2):
            plt.plot(trainingHistory.history[trainOutcome[title[position]][outcome]])
        plt.xlabel("epoch", fontsize=15)
        plt.ylabel(title[position], fontsize=15)
        plt.legend(['train', 'validation'], loc='upper left')

    plt.savefig("train History")


def main():

    model = InceptionResnetV2Model(shape,optimizer,activation,classes,loss).modelCreate()

    trainDataGen = loadImgGenFromDir(trainDir,(299,299),"categorical",batchSize,rotateAngle=60,widthShift=40,heightShift=40,horizon=True).loadByGenFlowFromDir()
    validDataGen = loadImgGenFromDir(validDir,(299,299),"categorical",batchSize).loadByGenFlowFromDir()

    initialModel = trainModel(epochs,validationSteps,stepsPerEpoch,trainDataGen,validDataGen)
    trainHistory = initialModel.startTrain(model)
    showTrainHistory(trainHistory)

if __name__ == "__main__":

    main()

