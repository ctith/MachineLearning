# Machine Learning

## Installer environnement de travail

### Pré-requis 
Java 7+ installé sur l'ordinateur

### Télécharger Anaconda 
https://anaconda.org/anaconda/python/files
- Windows 64-bits, python 2.7

### Télécharger Spark à la racine de C:/
http://spark.apache.org/downloads.html 
- Spark 2.1.0 (Déc 28 2016)
- Package type : Pre-built for Apache Hadoop 2.7 and later

### Télécharger winutils.exe et le mettre dans le dossier bin de Spark
https://github.com/steveloughran/winutils/blob/master/hadoop-2.6.0/bin/winutils.exe?raw=true

### Mettre Spark dans les [variables d'environnement](https://ss64.com/nt/set.html)
Dans le path système : 
```
C:\spark-2.3.0-bin-hadoop2.7\bin
```

Dans les variables utilisateurs :
```
setx SPARK_HOME C:\spark-2.3.0-bin-hadoop2.7\bin
setx HADOOP_HOME C:\spark-2.3.0-bin-hadoop2.7\bin
setx PYSPARK_DRIVER_PYTHON ipython
setx PYSPARK_DRIVER_PYTHON_OPTS notebook
```
