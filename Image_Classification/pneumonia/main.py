from model import InceptionResnetV2Model
from loadData import loadImgGenFromDir
from train import trainModel
from tensorflow.keras import optimizers
import os
import matplotlib.pyplot as plt

shape = (299,299,3)
learningRate = 3e-5
optimizer = optimizers.Adam(learningRate)
activation = "sigmoid"
classes = 1
loss = "binary_crossentropy"
baseDir = "/home/danny/Python_project/chest-xray-pneumonia/chest_xray/chest_xray"
trainDir = os.path.join(baseDir,"train1")
validDir = os.path.join(baseDir,"valid1")
batchSize = 20
epochs = 30
validationSteps = 1170/batchSize
stepsPerEpoch = (3512 /8)

def showTrainHistory(trainingHistory):

    plt.figure(figsize=(20, 10))
    plt.suptitle('Train History', fontsize=30)

    title = ["accuracy", "loss"]
    trainOutcome = {"accuracy": ["acc", "val_acc"], "loss": ["loss", "val_loss"]}

    for position in range(2):
        plt.subplot(1, 2, position + 1)
        plt.title(title[position])
        for outcome in range(2):
            plt.plot(trainingHistory.history[trainOutcome[position][outcome]])
        plt.xlabel("epoch", fontsize=15)
        plt.ylabel(title[position], fontsize=15)
        plt.legend(['train', 'validation'], loc='upper left')

    plt.savefig("train History")


def main():

    model = InceptionResnetV2Model(shape,optimizer,activation,classes,loss).modelCreate()

    trainDataGen = loadImgGenFromDir(trainDir,(299,299),"binary",batchSize=8,rotateAngle=30,horizon=True).loadByGenFlowFromDir()
    validDataGen = loadImgGenFromDir(validDir,(299,299),"binary",batchSize).loadByGenFlowFromDir()

    initialModel = trainModel(epochs,validationSteps,stepsPerEpoch,trainDataGen,validDataGen)
    trainHistory = initialModel.startTrain(model)
    showTrainHistory(trainHistory)

if __name__ == "__main__":
    main()
