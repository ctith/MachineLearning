# MLlib K-means

Si erreur 
```python
NameError: name 'sc' is not defined
```
Rajouter dans le script :
```python
from pyspark import SparkContext
sc = SparkContext("local", "App Name")
```

## EX0

Créer un fichier ex1kmeans.txt avec pour contenu 2,4,6,7,8,11,3

Lancer sur un IDE pyspark en input :
```python
#!/usr/bin/env python
from pyspark.mllib.clustering import KMeans
from numpy import array 
from math import sqrt

# charger les données
data = sc.textFile("ex1kmeans.txt")

#préparer les données
parsedData = data.map(lambda line:array([float(x) for x in line.split(',')]))

# créer le modèle
clusters = KMeans.train(parsedData,3)

# afficher les centres des clusters
print(clusters.clusterCenters)
```
Output :
```python
[array([ 2.,  4.,  6.,  7.,  8., 11., 13.])]
```
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2016_26_31-Ex0%20-%20MLlib%20K-means.png)

## EX1
Input :
```python
#!/usr/bin/env python
from pyspark.mllib.clustering import KMeans
from numpy import array 
from math import sqrt

from pyspark import SparkContext
sc = SparkContext("local", "App Name")

# charger les données
data = sc.textFile("data/mllib/kmeans_data.txt")

# préparer les données
splitedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

# créer le modèle
clusters = KMeans.train(splitedData , 3, maxIterations=10)

# afficher les centres des clusters
print(clusters.clusterCenters)
```
Output :
```python
[array([9.1, 9.1, 9.1]), array([0.05, 0.05, 0.05]), array([0.2, 0.2, 0.2])]
```

![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2016_18_25-Ex1%20-%20MLlib%20K-means.png)

## EX2
Input :
```python
#!/usr/bin/env python
from pyspark.mllib.clustering import KMeans
from numpy import array 
from math import sqrt

from pyspark import SparkContext
sc = SparkContext("local", "App Name")

# charger les données
data = sc.textFile("data/mllib/3D_spatial_network.txt")

#préparer les données
parsedData = data.map(lambda line: array([float(x) for x in line.split(',')]))

# créer le modèle
clusters = KMeans.train(parsedData, 3, maxIterations=20)

# afficher les centres des clusters
clusters.clusterCenters
```
Output :
```python
[array([9.90633422e+07, 9.68557343e+00, 5.70686553e+01, 2.11937929e+01]),
 array([3.43385461e+07, 9.81595312e+00, 5.71211049e+01, 1.97624911e+01]),
 array([1.38119067e+08, 9.75145608e+00, 5.70835976e+01, 2.54127126e+01])]
```
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2016_18_11-Ex2%20-%20MLlib%20K-means.png)
------------------------------

# MLlib FPGrowth
Input :
```python
#!/usr/bin/env python
from pyspark.mllib.fpm import FPGrowth

from pyspark import SparkContext
sc = SparkContext("local", "App Name")

# charger les données
data = sc.textFile("data/mllib/sample_fpgrowth.txt")

# préparer les données
splitedData= data.map(lambda line: line.strip().split(' '))

# appliquer FP-Growth
model = FPGrowth.train(splitedData, minSupport=0.2, numPartitions=10)
result = model.freqItemsets().collect()

# afficher le résultat
for item in result:
print(item)
```
Output :
```python
FreqItemset(items=[u'z'], freq=5)
FreqItemset(items=[u'x'], freq=4)
FreqItemset(items=[u'x', u'z'], freq=3)
FreqItemset(items=[u'y'], freq=3)
FreqItemset(items=[u'y', u'x'], freq=3)
FreqItemset(items=[u'y', u'x', u'z'], freq=3)
FreqItemset(items=[u'y', u'z'], freq=3)
FreqItemset(items=[u'r'], freq=3)
FreqItemset(items=[u'r', u'x'], freq=2)
FreqItemset(items=[u'r', u'z'], freq=2)
FreqItemset(items=[u's'], freq=3)
FreqItemset(items=[u's', u'y'], freq=2)
FreqItemset(items=[u's', u'y', u'x'], freq=2)
FreqItemset(items=[u's', u'y', u'x', u'z'], freq=2)
FreqItemset(items=[u's', u'y', u'z'], freq=2)
FreqItemset(items=[u's', u'x'], freq=3)
FreqItemset(items=[u's', u'x', u'z'], freq=2)
FreqItemset(items=[u's', u'z'], freq=2)
FreqItemset(items=[u't'], freq=3)
FreqItemset(items=[u't', u'y'], freq=3)
FreqItemset(items=[u't', u'y', u'x'], freq=3)
FreqItemset(items=[u't', u'y', u'x', u'z'], freq=3)
FreqItemset(items=[u't', u'y', u'z'], freq=3)
FreqItemset(items=[u't', u's'], freq=2)
FreqItemset(items=[u't', u's', u'y'], freq=2)
FreqItemset(items=[u't', u's', u'y', u'x'], freq=2)
FreqItemset(items=[u't', u's', u'y', u'x', u'z'], freq=2)
FreqItemset(items=[u't', u's', u'y', u'z'], freq=2)
FreqItemset(items=[u't', u's', u'x'], freq=2)
FreqItemset(items=[u't', u's', u'x', u'z'], freq=2)
FreqItemset(items=[u't', u's', u'z'], freq=2)
FreqItemset(items=[u't', u'x'], freq=3)
FreqItemset(items=[u't', u'x', u'z'], freq=3)
FreqItemset(items=[u't', u'z'], freq=3)
FreqItemset(items=[u'p'], freq=2)
FreqItemset(items=[u'p', u'r'], freq=2)
FreqItemset(items=[u'p', u'r', u'z'], freq=2)
FreqItemset(items=[u'p', u'z'], freq=2)
FreqItemset(items=[u'q'], freq=2)
FreqItemset(items=[u'q', u'y'], freq=2)
FreqItemset(items=[u'q', u'y', u'x'], freq=2)
FreqItemset(items=[u'q', u'y', u'x', u'z'], freq=2)
FreqItemset(items=[u'q', u'y', u'z'], freq=2)
FreqItemset(items=[u'q', u't'], freq=2)
FreqItemset(items=[u'q', u't', u'y'], freq=2)
FreqItemset(items=[u'q', u't', u'y', u'x'], freq=2)
FreqItemset(items=[u'q', u't', u'y', u'x', u'z'], freq=2)
FreqItemset(items=[u'q', u't', u'y', u'z'], freq=2)
FreqItemset(items=[u'q', u't', u'x'], freq=2)
FreqItemset(items=[u'q', u't', u'x', u'z'], freq=2)
FreqItemset(items=[u'q', u't', u'z'], freq=2)
FreqItemset(items=[u'q', u'x'], freq=2)
FreqItemset(items=[u'q', u'x', u'z'], freq=2)
FreqItemset(items=[u'q', u'z'], freq=2)
```
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2016_28_59-MLlib%20FPGrowth.png)
------------------------

# MLlib DecisionTree

## EX1
Input :
```python
#!/usr/bin/env python
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree 
from pyspark.mllib.util import MLUtils

from pyspark import SparkContext
sc = SparkContext("local", "App Name")

# charger les données
data = MLUtils.loadLibSVMFile(sc, 'data/mllib/sample_libsvm_data.txt').cache()

# appliquer les arbres de décisions
model = DecisionTree.trainClassifier(data, numClasses=2, categoricalFeaturesInfo={},impurity='gini', maxDepth=5)

# afficher le modèle
print(model.toDebugString())

# évaluer le résultat
predictions = model.predict(data.map(lambda x: x.features))
predictions.collect()

labelsAndPredictions = data.map(lambda lp: lp.label).zip(predictions)
labelsAndPredictions.collect()
trainErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(data.count())

print('Training Error = ' + str(trainErr))
print('Learned classification tree model:')
print(model)
```
Output :
```python
DecisionTreeModel classifier of depth 2 with 5 nodes
  If (feature 434 <= 70.5)
   If (feature 100 <= 193.5)
    Predict: 0.0
   Else (feature 100 > 193.5)
    Predict: 1.0
  Else (feature 434 > 70.5)
   Predict: 1.0

Training Error = 0.0
Learned classification tree model:
DecisionTreeModel classifier of depth 2 with 5 nodes
```
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2016_31_57-EX1%20-%20MLlib%20DecisionTree.png)

## EX2
Input :
```python
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

def mapping(line):
  line= line.replace('-1.000000', '0.000000')
  line= line.replace('-', '')
  return line

def toWriteFile(data):
  return ''.join(str(d) for d in data)
  data = sc.textFile("data/mllib/duke 2.csv")
  parsed=data.map(mapping)
  parsed.collect()
  lines=parsed.map(toWriteFile)
  lines.saveAsTextFile('ex1.txt')
  
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree 
from pyspark.mllib.util import MLUtils

from pyspark import SparkContext
sc = SparkContext("local", "App Name")

# charger les données
data = MLUtils.loadLibSVMFile(sc, '/data/mllib/ex1.txt/part-00000').cache()

# Appliquer les arbres de décisions
model = DecisionTree.trainClassifier(data, numClasses=2, categoricalFeaturesInfo={},impurity='gini', maxDepth=7)

# Afficher le modèle
print(model.toDebugString())
```
Output :
> Py4JJavaError: An error occurred while calling z:org.apache.spark.api.python.PythonRDD.collectAndServe.
: org.apache.hadoop.mapred.InvalidInputException: Input path does not exist: file:/data/mllib/ex1.txt/part-00000
![]()
---------------------

# MLlib Random Forest

## EX1
Input :
```python
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

from pyspark import SparkContext
sc = SparkContext("local", "App Name")

# Charger les données
data = MLUtils.loadLibSVMFile(sc, 'data/mllib/sample_libsvm_data.txt')

# Découper les données ensemble d’apprentissage et de test (70%,30%)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Appliquer les forêts aléatoires.
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, numTrees=3)

# Evaluer le modèle
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())

print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print(model.toDebugString())
```
Output :
```python
Test Error = 0.0357142857143
Learned classification forest model:
TreeEnsembleModel classifier with 3 trees

  Tree 0:
    If (feature 510 <= 2.5)
     If (feature 351 <= 1.0)
      Predict: 0.0
     Else (feature 351 > 1.0)
      Predict: 1.0
    Else (feature 510 > 2.5)
     Predict: 0.0
  Tree 1:
    If (feature 540 <= 10.0)
     Predict: 1.0
    Else (feature 540 > 10.0)
     Predict: 0.0
  Tree 2:
    If (feature 379 <= 11.5)
     If (feature 597 <= 15.0)
      Predict: 1.0
     Else (feature 597 > 15.0)
      Predict: 0.0
    Else (feature 379 > 11.5)
     Predict: 1.0
```
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2016_50_55-MLlib%20Random%20Forest%20-%20EX1.png)

## EX2
Input :
```python
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

from pyspark import SparkContext
sc = SparkContext("local", "App Name")

# préparer les données
def MapLine(line):
  val = [float(x) for x in line.split(';')]
  if(val[11]==5.0):
    return LabeledPoint(0, val[:10])
  else:
    return LabeledPoint(1, val[:10])
    
data = sc.textFile("/data/mllib/winequality-red.csv")
data = data.filter(lambda line: 'fixed acidity' not in line)
labelData = data.map(MapLine)
(trainingData, testData) = labelData.randomSplit([0.7, 0.3])
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, numTrees=3, impurity='gini')

# Afficher le modèle
print(model.toDebugString())

# Evaluer le modèle
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())

print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
```
Output :
> Py4JJavaError: An error occurred while calling o30.partitions.
: org.apache.hadoop.mapred.InvalidInputException: Input path does not exist: file:/data/mllib/winequality-red.csv

------------------
# MLlib Regression

Input :
```python
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel

from pyspark import SparkContext
sc = SparkContext("local", "App Name")

# Charger les données
data = MLUtils.loadLibSVMFile(sc, 'data/mllib/sample_libsvm_data.txt')

# Découper les données ensemble d’apprentissage et de test (70%,30%)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Appliquer les forêts aléatoires.
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={}, numTrees=3)

# Evaluer le modèle
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())

print('Test Error = ' + str(testErr))
print('Learned classification forest model:')
print(model.toDebugString())

# charger et préparer les données
def parsePoint(line):
  values = [float(x) for x in line.replace(',', ' ').split(' ')]
  return LabeledPoint(values[0], values[1:])

data = sc.textFile("data/mllib/ridge-data/lpsa.data")
parsedData = data.map(parsePoint)

# créer le modèle
model = LinearRegressionWithSGD.train(parsedData, iterations=100)

# évaluer le résultat
VP = parsedData.map(lambda p: (p.label, model.predict(p.features)))
MSE = VP.map(lambda (v, p): (v - p)**2) .reduce(lambda x, y: x + y) / data.count()
print("Mean Squared Error = " + str(MSE))

# Sauvegarder le modèle
model.save(sc, "target/tmp/pythonLinearRegressionWithSGDModel")
OurModel = LinearRegressionModel.load(sc,"target/tmp/pythonLinearRegressionWithSGDModel")
```
Output :
```python
Test Error = 0.0740740740741
Learned classification forest model:
TreeEnsembleModel classifier with 3 trees

  Tree 0:
    If (feature 433 <= 52.5)
     If (feature 539 <= 18.5)
      If (feature 152 <= 2.5)
       Predict: 1.0
      Else (feature 152 > 2.5)
       Predict: 0.0
     Else (feature 539 > 18.5)
      Predict: 0.0
    Else (feature 433 > 52.5)
     Predict: 1.0
  Tree 1:
    If (feature 328 <= 24.0)
     If (feature 511 <= 1.5)
      Predict: 1.0
     Else (feature 511 > 1.5)
      Predict: 0.0
    Else (feature 328 > 24.0)
     Predict: 0.0
  Tree 2:
    If (feature 407 <= 26.0)
     If (feature 212 <= 51.5)
      Predict: 1.0
     Else (feature 212 > 51.5)
      Predict: 0.0
    Else (feature 407 > 26.0)
     Predict: 1.0

Mean Squared Error = 6.207597210613579
```
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2016_59_27-MLlib%20Regression.png)

