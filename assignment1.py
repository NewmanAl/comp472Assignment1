# COMP472 Assignment 1
# Alexander Newman
# 27021747
# Oct 2020

from numpy import genfromtxt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

import matplotlib.pyplot as plt

from pathlib import Path

info1File = 'info_1.csv'
info2File = 'info_2.csv'

trainCsvFile1 = 'train_1.csv'
trainCsvFile2 = 'train_2.csv'

valCsvFile1 = 'val_1.csv'
valCsvFile2 = 'val_2.csv'

testCsvFile1 = 'test_with_label_1.csv'
testCsvFile2 = 'test_with_label_2.csv'

info1Legend = genfromtxt(info1File, delimiter=",", dtype='unicode')[1:,-1]
info2Legend = genfromtxt(info2File, delimiter=",", dtype='unicode')[1:,-1]


def main():
  
  trainingFile1 = trainCsvFile1
  trainingFile2 = trainCsvFile2
  testFile1 = testCsvFile1
  testFile2 = testCsvFile2
  
  #Gaussian Naive Bayes
  clf = GaussianNB()
  runModel(trainingFile1,testFile1,"GNB-DS1", info1Legend, clf)
  clf = GaussianNB()
  runModel(trainingFile2,testFile2,"GNB-DS2", info2Legend, clf)
  
  #Baseline Decision Tree
  clf = DecisionTreeClassifier(criterion='entropy')
  runModel(trainingFile1,testFile1,"Base-DT-DS1", info1Legend, clf)
  clf = DecisionTreeClassifier(criterion='entropy')
  runModel(trainingFile2,testFile2,"Base-DT-DS2", info2Legend, clf)
  
  #Best Decision Tree
  clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_impurity_decrease=0.0, class_weight=None)
  runModel(trainingFile1,testFile1,"Best-DT-DS1", info1Legend, clf)
  clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_impurity_decrease=0.0, class_weight=None)
  runModel(trainingFile2,testFile2,"Best-DT-DS2", info2Legend, clf)
  
  #Perceptron
  clf = Perceptron()
  runModel(trainingFile1,testFile1,"PER-DS1", info1Legend, clf)
  clf = Perceptron()
  runModel(trainingFile2,testFile2,"PER-DS2", info2Legend, clf)
  
  #Baseline Multi-Layered Perceptron
  clf = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic')
  runModel(trainingFile1,testFile1,"Base-MLP-DS1", info1Legend, clf)
  clf = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic')
  runModel(trainingFile2,testFile2,"Base-MLP-DS2", info2Legend, clf)
  
  #Best Multi-Layered Perceptron
  clf = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam')
  runModel(trainingFile1,testFile1,"Best-MLP-DS1", info1Legend, clf)
  clf = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='adam')
  runModel(trainingFile2,testFile2,"Best-MLP-DS2", info2Legend, clf)
  

def runModel(trainFile, testFile, modelName, alphaLabels, clf):
  print(modelName.center(30))
  print("==============================")
  
  features, labels = readCsvData(trainFile)

  print("Fitting " + modelName + " model")
  clf.fit(features,labels)

  validationOutput = []
  validationFeatures, validationLabels = readCsvData(testFile)

  print("Generating output using " + modelName)
  for row in validationFeatures:
    validationOutput.append(clf.predict([row])[0])

  handleModelResults(validationOutput, validationLabels, modelName, alphaLabels)
  print()

def handleModelResults(predictedOutput, expectedOutput, modelName, labels):
  print("Calculating metrics for "+modelName)
  confMatrix, \
  precision, \
  recall, \
  f1score, \
  accuracy, \
  macroAvgF1, \
  weightedAvgF1 = calculateMetrics(expectedOutput, predictedOutput, list(range(0,len(labels))))
  
  print("Generating diagrams for " + modelName)
  generateDiagrams(labels, modelName, "diagrams", confMatrix, precision, recall, f1score, accuracy, macroAvgF1, weightedAvgF1) 
  
  print("Writting results to output/" + modelName + ".csv")
  writeResultsToFile('output/'+modelName+'.csv', predictedOutput, confMatrix, precision, recall, f1score, accuracy, macroAvgF1, weightedAvgF1)

def calculateMetrics(expectedOutput, predictedOutput, labels):
  confMatrix = metrics.confusion_matrix(expectedOutput, predictedOutput)
  
  precision, recall, fscore, support = metrics.precision_recall_fscore_support(expectedOutput, predictedOutput, labels=labels)
  
  accuracy = metrics.accuracy_score(expectedOutput, predictedOutput)
  
  macroAvgF1 = metrics.f1_score(expectedOutput, predictedOutput, average='macro')
  
  weightedAvgF1 = metrics.f1_score(expectedOutput, predictedOutput, average='weighted')
  
  return confMatrix, \
         precision, \
         recall, \
         fscore, \
         accuracy, \
         macroAvgF1, \
         weightedAvgF1

def writeResultsToFile(fileName,
                       predictedOutput,
                       confusionMatrix,
                       precision,
                       recall,
                       f1Score,
                       accuracy,
                       macroAvgF1,
                       weightedAvgF1):
  
  # ensure output directory exists
  # (assume paths separated by '/')
  # No Windows style paths here... :(
  directoryComponents = fileName.split('/')[:-1]
  if len(directoryComponents) > 0:
    directory = '/'.join(directoryComponents)
    Path(directory).mkdir(parents=True, exist_ok=True)
  
  with open(fileName, 'w') as f:
    i = 1
    # predicted class for each instance
    for row in predictedOutput:
        f.write('{0},{1:g}\n'.format(i,row))
        i+=1
        
    #confusion matrix
    for row in confusionMatrix:
      i = 1
      for col in row:
        f.write('{0:g}'.format(col))
        if i < row.size:
          f.write(',')
        i+=1
      f.write('\n')
      
    #precision
    i = 1
    f.write('precision,')
    for val in precision:
      f.write('{0:f}'.format(val))
      if i < precision.size:
        f.write(',')
      i+=1
    f.write('\n')
    
    #recall
    i = 1
    f.write('recall,')
    for val in recall:
      f.write('{0:f}'.format(val))
      if i < recall.size:
        f.write(',')
      i+=1
    f.write('\n')
    
    #f1 score
    i = 1
    f.write('f1 score,')
    for val in f1Score:
      f.write('{0:f}'.format(val))
      if i < f1Score.size:
        f.write(',')
      i+=1
    f.write('\n')
    
    #accuracy
    f.write('accuracy,{0:f}\n'.format(accuracy))
    
    #macro-average f1
    f.write('macro-average f1,{0:f}\n'.format(macroAvgF1))
    
    #weighted-average f1
    f.write('weighted-average f1,{0:f}'.format(weightedAvgF1))

def generateDiagrams(valueLegend,
                     modelName,
                     directoryName,
                     confusionMatrix,
                     precision,
                     recall,
                     f1Score,
                     accuracy,
                     macroAvgF1,
                     weightedAvgF1):
                     
  # ensure output directory exists
  Path(directoryName).mkdir(parents=True, exist_ok=True)
                     
  import plotConfusion
  plotConfusion.plot_confusion_matrix(confusionMatrix, \
                                      valueLegend, \
                                      normalize=False, \
                                      title=modelName+" Confusion Matrix", \
                                      saveToFile=True, \
                                      saveFileName=directoryName+"/"+modelName+"_confusionMatrix.png")
  
  plt.bar(valueLegend, precision)
  plt.title(modelName+" Precision")
  #plt.show()
  plt.savefig(directoryName+"/"+modelName+"_precision.png")
  plt.close()
  
  plt.bar(valueLegend, recall)
  plt.title(modelName+" Recall")
  #plt.show()
  plt.savefig(directoryName+"/"+modelName+"_recall.png")
  plt.close()
  
  plt.bar(valueLegend, f1Score)
  plt.title(modelName+" f1 Score")
  #plt.show()
  plt.savefig(directoryName+"/"+modelName+"_f1Score.png")
  plt.close()
  
  plt.bar(["accuracy", "macro-average f1", "weighted-average f1"], [accuracy, macroAvgF1, weightedAvgF1])
  plt.title(modelName)
  for index, value in enumerate([accuracy, macroAvgF1, weightedAvgF1]):
    plt.text(index, value + 0.003, '{0:.4f}'.format(value), horizontalalignment='center')
  #plt.show()
  plt.savefig(directoryName+"/"+modelName+"_accuracyMacroAvgWeightedAvg.png")
  plt.close()
  
# Reads the csv data file, and returns two numpy arrays
# The first numpy array contains the features
# The second numpy array contains the labels
def readCsvData(fileName):
  data = genfromtxt(fileName, delimiter=',')

  return data[:,:-1], data[:,-1]

if __name__ == '__main__':
  main()
