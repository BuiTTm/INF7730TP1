# MACHINE LEARNING MODEL

## Détails du travail
* Le code a été executé avec Python 3.7.11

Le code est composé de 6 fichiers:
* feature_extraction.py
* preprocessing.py
* comparaison_full_features.py
* selection_information_gain.py
* selection_chi2.py
* model_learning.py

Le fichier model_learning.py joue comme titre de librairie qui sont utilisé pour effectuer l'initialisation des modèles d'apprentissage automatique (DecisionTree, RandomForest, Bagging, ADA, NaivesBayes). Considérant les procédés sont relativement similaire pour l'analyse de tous les features, la sélection avec gain d'information et la sélection chi2, cette approche a été choisie.

# Features extraction

Dans cette partie, le dataset de Social Honeypot a été utilisé afin de réalisé l'ensemble de données nécessaire afin de créer le modèle d'apprentissage automatique en mesure d'identifier des usagers Spammer. Pour ce faire,
les features qui seront extraites sont:
* La longueur du nom de l'usager (LengthOfScreenName)
* La longévité du compte (AccountLongevity)
* Nombre de following (NumberOfFollowings)
* Nombre de followers (NumberOfFollowers)
* L'écart type des IDs followings unique (StdDevSeriesUniqueFollowing)
* Ratio following/followers (FonF_ratio)
* Nombre de tweet total envoyés (NumberOfTweets)
* Nombre moyen de tweet envoyé par jour (TweetSentPerDay)
* Nombre de tweets totals envoyé sur la longévité du compte (AvgTweetSentWhileActive)

# Preprocessing 

# Comparaison full features

# Selection information gain

# Selection chi2
