#Tache 1: Comparaison full features des algorithmes
#Auteur: Bui Trung Tin Michel

#Import of libraries
#Voir fichier model_learning.py
from model_learning import ModelLearning

#Constant values
FOLDER_LOCATION = '/Cleaned_Data'
DATA_LOCATION = f'.{FOLDER_LOCATION}/features_cleaned.csv'
#To create a real csv file of normalized dataframe
SCALED_DF_TO_CSV = False


        
if __name__ == "__main__":
    #Initialiazing object from model_learning class
    ml_obj = ModelLearning(DATA_LOCATION)
    
    #Normalizing the data using z-score
    ml_obj.normalize_data()
    
    #Creating training set and testing set
    ml_obj.split_dataset()
    
    #Initializing ML models
    ml_obj.decisiontree_init()
    ml_obj.randomforest_init()
    ml_obj.bagging_init()
    ml_obj.ada_init()
    ml_obj.naivebayes_init()
    
    #Calculating accuracy of models
    ml_obj.display_accuracy()
    
    #Creating confusion matrix
    ml_obj.confusion_matrix()
    
    #Calculating f-scores
    ml_obj.f_score()
    ml_obj.roc('./Figures/ROC_all_features')

    if SCALED_DF_TO_CSV:
        ml_obj.to_csv('normalized_dataset.csv')



