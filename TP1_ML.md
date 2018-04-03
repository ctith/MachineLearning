# MLlib K-means

```python
#!/usr/bin/env python
from pyspark.mllib.clustering import KMeans
from numpy import array 
from math import sqrt

from pyspark import SparkContext
sc = SparkContext("local", "App Name", pyFiles=['MyFile.py', 'lib.zip', 'app.egg'])

# charger les données
data = sc.textFile("data/mllib/kmeans_data.txt")

#preéparer les donneées
splitedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

# créer le modèle
clusters = KMeans.train(splitedData , 3, maxIterations=10)

# afficher les centres des clusters
print(clusters.clusterCenters)
```