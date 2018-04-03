# Machine Learning

## Installer environnement de travail

### Pré-requis 
Java 7+ installé sur l'ordinateur

### Télécharger Gow
Gow permet d'utiliser les commandes linux sur windows
https://github.com/bmatzelle/gow/releases/download/v0.8.0/Gow-0.8.0.exe

### Télécharger Anaconda 
https://www.anaconda.com/download/ 

- Windows 64-bits, python **2.7**

[Tuto installation](https://medium.com/@GalarnykMichael/install-python-on-windows-anaconda-c63c7c3d1444)

### Télécharger Spark à la racine de C:/
http://spark.apache.org/downloads.html 
- Spark 2.3.0 (prendre la dernière version ou la version N-1 + stable proposée sur le site)
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

### Installer Pyspark
Ouvrir la console **Anaconda prompt** et faire
```
pip install pyspark
```
