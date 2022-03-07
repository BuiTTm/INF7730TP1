#Tache 2A: Selection des meilleurs features selon le gain information
#Auteur: Bui Trung Tin Michel BUIT31059004

#Re-use of class previously created
from model_learning import ModelLearning
import pandas as pd
#For mutual info classif
from sklearn.feature_selection import mutual_info_classif

#Constant values
FOLDER_LOCATION = '/Cleaned_Data'
DATA_LOCATION = f'.{FOLDER_LOCATION}/features_cleaned.csv'
SCALED_DF_TO_CSV = False

if __name__ == "__main__":
    #Initialiazing object from model_learning class
    ml_select_feat_obj = ModelLearning(DATA_LOCATION)
    
    #Normalizing the data using z-score and extract the scaled dataframe
    x_scaled, y = ml_select_feat_obj.normalize_data(True)
    
    #Inputting the scaled dataframe to obtain best features
    data = mutual_info_classif(x_scaled, y, random_state = 66).reshape(-1, 1)
    fcoeff = pd.DataFrame(data, columns=['Coeff'], index=x_scaled.columns)
    
    #Only take 7 top features
    feat_top_7= fcoeff.sort_values(by='Coeff', ascending=False)[:7]
    print(50*'=')
    print(f'Top 7 features: \n {feat_top_7}')
    print(50*'=')
    
    #Deploy models with the best 7 features selected
    cols_top7= feat_top_7.reset_index()["index"].tolist()
    print(f'{cols_top7}')
    print(50*'=')
    ml_select_feat_obj.select_cols(cols_top7)
    
    #Creating training set and testing set
    ml_select_feat_obj.split_dataset()
    
    #Initialize ML models
    ml_select_feat_obj.decisiontree_init()
    ml_select_feat_obj.randomforest_init()
    ml_select_feat_obj.bagging_init()
    ml_select_feat_obj.ada_init()
    ml_select_feat_obj.naivebayes_init()
    
    #Calculating accuracy of models
    ml_select_feat_obj.display_accuracy()
    
    #Creating confusion matrix
    ml_select_feat_obj.confusion_matrix()

    #Calculating f-scores
    ml_select_feat_obj.f_score()
    ml_select_feat_obj.roc('ROC_best_features_info_gain')        
    ml_select_feat_obj.to_csv('normalized_dataset_info_gain.csv')
    
    if SCALED_DF_TO_CSV:
        ml_select_feat_obj.to_csv('normalized_dataset_inf_gain.csv')


