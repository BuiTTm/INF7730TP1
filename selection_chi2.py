#Tache 2B: Selection des meilleurs features selon le chi2
#Auteur: Bui Trung Tin Michel

#Re-use of class previously created
from model_learning import ModelLearning
import pandas as pd
#For chi2
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


#Constant values
FOLDER_LOCATION = '/Cleaned_Data'
DATA_LOCATION = f'.{FOLDER_LOCATION}/features_cleaned.csv'
SCALED_DF_TO_CSV = False

if __name__ == "__main__":
    #Initialiazing object from model_learning class
    ml_select_feat_obj = ModelLearning(DATA_LOCATION)
    
    #Extract necessary dataset for chi2 fitting
    x = ml_select_feat_obj.get_x()
    y = ml_select_feat_obj.get_y()
    
    #Chi2 selection
    chi2score = chi2(x, y)
    chi2top7 = SelectKBest(chi2, k=7).fit(x,y)
    chi2results = pd.Series(chi2top7.get_support(), index=x.columns)
    feat_top_7 = chi2results.sort_values(ascending=False)
    feat_top_7 = feat_top_7.iloc[feat_top_7.values == True].reset_index()
    print(50*'=')
    print(f'Top 7 features: \n {feat_top_7}')
    print(50*'=')

    #Selection of DataFrame with the 7 selected features
    cols_top7= feat_top_7.reset_index()["index"].tolist()
    print(f'{cols_top7}')
    
    #Scale data
    ml_select_feat_obj.normalize_data()
    #Deploy models with the best 7 features selected
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
    ml_select_feat_obj.roc('ROC_best_features_chi2')        
    
    if SCALED_DF_TO_CSV:
        ml_select_feat_obj.to_csv('normalized_dataset_chi2.csv')
