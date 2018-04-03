# MLlib K-means

## EX0

Créer un fichier ex1kmeans.txt avec pour contenu 2,4,6,7,8,11,3

Lancer sur un IDE pyspark
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
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2016_26_31-Ex0%20-%20MLlib%20K-means.png)

## EX1
```python
#!/usr/bin/env python
from pyspark.mllib.clustering import KMeans
from numpy import array 
from math import sqrt

from pyspark import SparkContext
sc = SparkContext("local", "App Name")

# charger les données
data = sc.textFile("data/mllib/kmeans_data.txt")

#preéparer les donneées
splitedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

# créer le modèle
clusters = KMeans.train(splitedData , 3, maxIterations=10)

# afficher les centres des clusters
print(clusters.clusterCenters)
```
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2016_18_25-Ex1%20-%20MLlib%20K-means.png)

## EX2
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
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2016_18_11-Ex2%20-%20MLlib%20K-means.png)
------------------------------

# MLlib FPGrowth
```python
#!/usr/bin/env python
from pyspark.mllib.fpm import FPGrowth

from pyspark import SparkContext
sc = SparkContext("local", "App Name")

# charger les données
data = sc.textFile("data/mllib/sample_fpgrowth.txt")

#Préparer les données
splitedData= data.map(lambda line: line.strip().split(' '))

# Appliquer FP-Growth
model = FPGrowth.train(splitedData, minSupport=0.2, numPartitions=10)
result = model.freqItemsets().collect()

#Afficher le résultat
for item in result:
print(item)
```

------------------------

# MLlib DecisionTree

## EX1
```python
#!/usr/bin/env python
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree 
from pyspark.mllib.util import MLUtils

from pyspark import SparkContext
sc = SparkContext("local", "App Name")

# charger les données
data = MLUtils.loadLibSVMFile(sc, 'data/mllib/sample_libsvm_data.txt').cache()

# Appliquer les arbres de décisions
model = DecisionTree.trainClassifier(data, numClasses=2, categoricalFeaturesInfo={},impurity='gini', maxDepth=5)

#Afficher le modèle
print(model.toDebugString())

# Evaluer le résultat
predictions = model.predict(data.map(lambda x: x.features))
predictions.collect()

labelsAndPredictions = data.map(lambda lp: lp.label).zip(predictions)
labelsAndPredictions.collect()
trainErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() /
float(data.count())

print('Training Error = ' + str(trainErr))
print('Learned classification tree model:')
print(model)
```

## EX2
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

# charger les données
data = MLUtils.loadLibSVMFile(sc, '/data/mllib/ex1.txt/part-00000').cache()

# Appliquer les arbres de décisions
model = DecisionTree.trainClassifier(data, numClasses=2,
categoricalFeaturesInfo={},impurity='gini', maxDepth=7)

#Afficher le modèle
print(model.toDebugString())
```
---------------------

