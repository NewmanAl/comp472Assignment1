import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

info1File = 'info_1.csv'
info2File = 'info_2.csv'

trainCsvFile1 = 'train_1.csv'
trainCsvFile2 = 'train_2.csv'

testCsvFile1 = 'test_with_label_1.csv'
testCsvFile2 = 'test_with_label_2.csv'

def plotLabelFrequency(trainDataFile, infoFile, fileName):

  trainLabels = genfromtxt(trainDataFile, delimiter=',')[:,-1] 
  
  infoLegend = genfromtxt(infoFile, delimiter=',', dtype='unicode')[1:,-1]
  
  uniqueLabels, countLabels = np.unique(trainLabels, return_counts=True)
  
  plt.bar(infoLegend, countLabels)
  plt.title(trainDataFile+" contents")
  #plt.show()
  plt.savefig("diagrams/" + fileName)
  plt.close()


if __name__ == '__main__':
  plotLabelFrequency(trainCsvFile1, info1File, "train1Data.png")
  plotLabelFrequency(trainCsvFile2, info2File, "train2Data.png")
  plotLabelFrequency(testCsvFile1, info1File, "test1Data.png")
  plotLabelFrequency(testCsvFile2, info2File, "test2Data.png")