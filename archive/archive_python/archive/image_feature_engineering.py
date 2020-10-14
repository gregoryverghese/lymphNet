import numpy as np
import cv2
from scipy import ndimage as nd
from skimage.filters import prewitt, sobel, scharr, roberts
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class GaborFilters():
    def __init__(self, img):
        self.img = img


    def getGabor(self, ksize, sigma, theta, lamda, gamma, l, ktype):

        kernel=cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, l, ktype=ktype)
        fimg = cv2.filter2D(self.img.reshape(-1), cv2.CV_8UC3, kernel)
        filteredImage=fimg.reshape(-1)

        return filteredImage


    def getGabors(self):

        ksize=5
        thetas = list(map(lambda x: x*0.25*np.pi, [1, 2]))
        gabors=[]

        for theta in thetas:
            for sigma in (1,3):
                for lamda in np.arange(0, np.pi, np.pi*0.25):
                    for gamma in (0.05, 0.5):
                        gabor = self.getGabor(ksize, sigma, theta, lamda, gamma, 0, cv2.CV_32F)
                        gabors.append(gabor)

        return gabors

class FeatureEngineering():

    def __init__(self, img):
        self.img = img


    def getCanny(self):

        edges = cv2.Canny(self.img, 100, 200)
        edges1 = edges.reshape(-1)

        return edges1


    def getRoberts(self):

        edgeroberts = roberts(self.img)
        edgeroberts1 = edgeroberts.reshape(-1)


        return edgeroberts1


    def getSobel(self):

        edgeSobel = sobel(self.img)
        edgeSobel1 = edgeSobel.reshape(-1)

        return edgeSobel1


    def getScharr(self):

        edgeScharr = scharr(self.img)
        edgeScharr1 = edgeScharr.reshape(-1)

        return edgeScharr1


    def getPrewitt(self):

        edgePrewitt = prewitt(self.img)
        edgePrewitt1 = edgePrewitt.reshape(-1)

        return edgePrewitt1


    def getGaussian(self, sigma):

        gaussianImg = nd.gaussian_filter(self.img, sigma=sigma)
        gaussianImg1 = gaussianImg.reshape(-1)

        return gaussianImg1


    def getMedian(self, size=3):

        medianImg = nd.median_filter(self.img, size=size)
        medianImg1 = medianImg.reshape(-1)

        return medianImg1


    def getLabels(self, mask):

        labels = mask.reshape(-1)

        return labels


    def getHOG(self, image, orientations=9, pixpercell=(3,3), cellsperblock=(2,2), sqrt=True, bNorm='L1', visualize=True):

        H, hogImage = hog(image, orientations=orientations, pixels_per_cell=pixpercell, cells_per_block=cellsperblock, transform_sqrt=sqrt, block_norm=bNorm, visualize=visualize)

        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")

        return H, hogImage


    def getLocalBinaryPattern(self, img, lbpParams):

        lbp = local_binary_pattern(img, **lbpParams)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, lbpParams['P'] + 3), range=(0, lbpParams['P'] + 2))

        hist=hist.astype('float')
        hist /= (hist.sum())

        return lbp, hist
