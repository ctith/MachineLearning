# Installer l'environnement de travail

## Pré-requis 
Java 7+ installé sur l'ordinateur

## Télécharger Gow
Gow permet d'utiliser les commandes linux sur windows
- https://github.com/bmatzelle/gow/releases/download/v0.8.0/Gow-0.8.0.exe

## Télécharger Anaconda 

[Tuto installation](https://medium.com/@GalarnykMichael/install-python-on-windows-anaconda-c63c7c3d1444)

- https://www.anaconda.com/download/ 

- Windows 64-bits, python **2.7**

## Télécharger Spark à la racine de C:/
http://spark.apache.org/downloads.html 
- Spark 2.3.0 (prendre la dernière version ou la version N-1 + stable proposée sur le site)
- Package type : Pre-built for Apache Hadoop 2.7 and later

## Télécharger winutils.exe et le mettre dans le dossier bin de Spark
https://github.com/steveloughran/winutils/blob/master/hadoop-2.6.0/bin/winutils.exe?raw=true

## Mettre Spark dans les [variables d'environnement](https://ss64.com/nt/set.html)
Dans les variables utilisateurs :
```
SPARK_HOME C:\spark-2.3.0-bin-hadoop2.7\bin
HADOOP_HOME C:\spark-2.3.0-bin-hadoop2.7\bin
PYSPARK_DRIVER_PYTHON ipython
PYSPARK_DRIVER_PYTHON_OPTS notebook
```
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2015_02_30-Variables%20d%E2%80%99environnement.png)

Dans le path système : 
```
C:\spark-2.3.0-bin-hadoop2.7\bin
```
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2015_02_46-Modifier%20la%20variable%20d'environnement.png)

## Installer Pyspark
Ouvrir la console **Anaconda prompt** et faire
```
pip install pyspark
```
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2015_00_53-Anaconda%20Prompt.png)

# Lancer les IDE

## Anaconda navigator
Launche Jupyter and Spyder
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2015_24_50-ctith_MachineLearning_%20PySpark.png)

## Jupyter
Créer un nouveau dossier qui sera notre workspace (new folder) 
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2015_41_26-Home.png)

puis créer un nouveau fichier (new file) pour écrire notre script python.
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2015_23_03-Untitled.png)

## Spyder
![](https://github.com/ctith/MachineLearning/blob/master/ml_screenshot/2018-04-03%2015_24_06-Python%20Programming%20Guide%20-%20Spark%200.9.1%20Documentation.png)
