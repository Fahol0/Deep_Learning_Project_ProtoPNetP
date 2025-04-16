# Deep_Learning_Project_ProtoPNetP

## Classification d'images CIFAR-10 - MLRF
Projet réalisé dans le cadre du cours de Machine Learning pour la Reconnaissance de Formes (MLRF) par Paul Giraud et Fabien Holard.

Ce projet explore différentes méthodes d'extraction de caractéristiques et plusieurs algorithmes de classification classiques afin de catégoriser les images du dataset CIFAR-10.

### Objectifs
Implémenter des techniques d'extraction de features : Flatten, Histogram of Color (HoC), Histogram of Oriented Gradients (HOG)

Comparer plusieurs modèles de classification classiques : Logistic Regression, Random Forest, Support Vector Machine, Stochastic Gradient Descent

Évaluer les performances selon différentes combinaisons feature/model

Obtenir une précision compétitive sans recourir au deep learning

### Technologies utilisées
Python 3.10.12

NumPy 1.21.0

OpenCV 4.5.3

scikit-learn 0.24.2

Matplotlib 3.4.2

Jupyter Notebook

### Dataset
CIFAR-10 : 60,000 images en couleur de taille 32x32 réparties en 10 classes.

Séparation : 50,000 pour l'entraînement, 10,000 pour le test.

Classes : avion, auto, oiseau, chat, cerf, chien, grenouille, cheval, navire, camion.

### Prétraitement
Chargement & reshape des images

Normalisation des pixels (valeurs entre 0 et 1)

Extraction de features via différentes techniques

Séparation train/test : 80/20

### Méthodes de classification

Modèle  	     |       Feature Extraction  |	Accuracy
--------------------------------------------------------
Random Forest    |    	HOG           | 	      50.13 % \
SVM	           |       HOG	      |            62.08 % \
Logistic Regression	|  HOG	         |         52.14 % \
SVM	         |         Flatten	      |        54.36 % \
NB : Les modèles avec HOG sont ceux qui performent le mieux.

### Lancer le projet
Cloner le repo / récupérer les fichiers

Installer les dépendances :

pip install numpy opencv-python scikit-learn matplotlib

Ouvrir le notebook :

jupyter notebook notebook.ipynb
### Améliorations possibles
Intégration de modèles de Deep Learning (CNNs)

Combinaisons de features avancées (ex : PCA + HOG)

Grid Search plus poussé pour l’optimisation des hyperparamètres

Analyse plus fine des erreurs via la matrice de confusion
