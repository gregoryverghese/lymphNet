import numpy as np
import matplotlib.pyplot as plt

class Decay():
    def plot(self, epochs, title='Learning rate schedule'):
        epochs = np.arange(0, epochs)
        lrs = [self(i) for i in epochs]
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")


class PieceWiseDecay(Decay):
    def __init__(self,  initLr=0.01, factor=0.5, decayTime=10):
        self.initLr = initLr
        self.factor = factor
        self.decayTime = decayTime

    def __call__(self, epoch):
        exponent = np.floor((1+epoch)/self.decayTime)
        lr = self.initLr * np.power(self.factor, exponent)
        return float(lr)


class PolynomialDecay(Decay):
    def __init__(self, maxEpochs=100, initLr=0.01, power=1):
        self.maxEpochs = maxEpochs
        self.initLr = initLr
        self.power = power

    def __call__(self, epoch):
        decay = 1 - np.power(epoch / float(self.maxEpochs), self.power)
        lr = self.initLr * decay
        return  float(lr)


