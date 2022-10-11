#Features extraction 
#Auteur: Bui Trung Tin Michel 

#Import of libraries
import numpy as np
import pandas as pd

FOLDER_LOCATION = './Cleaned_Data'
FEATURE_RAW_DATA_LOCATION = f'{FOLDER_LOCATION}/features_raw.csv'
FEATURE_CLEANED_DATA_LOCATION = f'{FOLDER_LOCATION}/features_cleaned.csv'

class DataCleaner():
    
    def __init__(self, filename) -> None:
        self.df = pd.read_csv(filename)
        
        
    def fill_missing_data(self):
        """Function to fill out
        missing data in the final dataset
        """
        #Column with missing data
        cols = ['FonF_ratio', 'TweetSentPerDay', 'AvgTweetSentWhileActive',
                'TweetUrlOnTweet', 'AvgURLPerTweet', 'MentionsPerTweet', 
                'AvgTimeBetween', 'MaxTimeBetween' ]
        
        #Fillout missing values with median
        print(50*'=')
        print('Nombre de valeurs contenant des np nan avant remplacement par median:')
        print(self.df.isna().sum()) 
        self.df[cols] = self.df[cols].fillna(self.df[cols].median())
        print(50*'=')
        print('Nombre de valeurs contenant des np nan apres remplacement par median:')
        print(self.df.isna().sum()) 
    
    
    def replace_inf_data(self):
        """Function to replace np inf caused by missing values to obtain
        ratio on certain features
        """
        print(50*'*')
        print('Nombre de valeurs contenant des np inf avant procedure:')
        print(self.df.iloc[self.df.values==np.inf].sum())
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(50*'*')
        print('Nombre de valeurs contenant des np inf apres procedure:')
        print(self.df.iloc[self.df.values==np.inf].sum())

    def to_csv(self, filename):
        self.df.to_csv(filename)

#Init Datacleaner class
df_obj = DataCleaner(FEATURE_RAW_DATA_LOCATION)
#Replace np inf values
df_obj.replace_inf_data()
#Replace np Nan values
df_obj.fill_missing_data()
#Final data csv file
df_obj.to_csv(FEATURE_CLEANED_DATA_LOCATION)
