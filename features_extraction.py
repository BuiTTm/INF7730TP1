#Fichier de pre-traitement 
#Auteur: Bui Trung Tin Michel 

#Chargement de librairie
import pandas as pd
import pickle as pck
import numpy as np


#import sklearn.preprocessing import Imputer
#Constant strings
SPAMMER = 'content_polluters'
LEGIT_USER = 'legitimate_users'

class FeaturesExtracter():
    
    def __init__(self) -> None:
        #Importation des données en CSV
        self.df_dict = {}
        self.merged_df_dict = {}   
        self.i = 1     

    def create_filenames(self):
        """Function to create filenames dictionary with 
        corresponding columns names
        """
        #Noms des colonnes
        user_cols = ['UserID', 'CreatedAt', 'CollectedAt', 
            'NumberOfFollowings', 'NumberOfFollowers', 
            'NumberOfTweets', 'LengthOfScreenName', 
            'LengthOfDescriptionInUserProfile']
        follow_cols = ['UserID', 'SeriesOfNumberofFollowings']
        tweet_cols = ['UserID', 'TweetID', 'Tweet', 'CreatedAt']
        
        #Dictionnaire des fichiers avec noms de colonnes
        self.files = {
            SPAMMER: user_cols, 
            f'{SPAMMER}_followings': follow_cols, 
            f'{SPAMMER}_tweets': tweet_cols,
            LEGIT_USER: user_cols, 
            f'{LEGIT_USER}_followings': follow_cols, 
            f'{LEGIT_USER}_tweets': tweet_cols
        }
        
        
    def read_raw_csv(self):
        """Read raw csv files from filename dictionary into
        DataFrame type and it will also display missing values
        and drop duplicates
        """
        #Lecture des fichiers textes pour les convertir en DataFrame
        for file, cols in self.files.items():
            self.df_dict[file] = pd.read_csv(f'./social_honeypot_icwsm_2011/{file}.txt', sep='\t', names=cols, index_col='UserID')

            print(50*'=')
            print(f'Nom de fichier:{file}')
            print(f'Valeurs manquantes:{self.df_dict[file].isna().sum()}')
            print(f'Valeurs=\n{self.df_dict[file][self.df_dict[file].isna().any(axis=1)]}')
            print(f'Size of dataframe before drop duplicate={self.df_dict[file].shape}')
            self.df_dict[file].drop_duplicates(inplace=True)
            print(f'Size of dataframe after drop duplicate={self.df_dict[file].shape}')
            self.df_dict[file].dropna(inplace=True)
            print(f'Size of dataframe after drop NA={self.df_dict[file].shape}')

            
    def verify_redundant_users(self):
        """Validate if there are common users between legit users 
        list and spammers list.
        """
        matched_users = pd.merge(self.df_dict[SPAMMER], self.df_dict[LEGIT_USER])
        print(50*'=')
        print(f'MATCHED USERS={len(matched_users)}')
        print(50*'=')


    def mark_spammer(self):
        """Function to mark spammer from list of users with
        a boolean.
        """
        #Create columns indicating if users is a spammer or not
        for user in [SPAMMER, LEGIT_USER]:   
            self.merged_df_dict[user] = self.df_dict[user]\
                .merge(self.df_dict[f'{user}_followings'], on='UserID') 
            if user is SPAMMER:
                self.merged_df_dict[user]['SpammerBoolean']=1
            elif user is LEGIT_USER:
                self.merged_df_dict[user]['SpammerBoolean']=0
                
            #Uncomment to save intermediary files
            #self.merged_df_dict[user].to_csv(f'./Cleaned_Data/{user}.csv')
            
    def create_main_dfs(self):
        #Merge users into one dataframe
        self.df = pd.concat([self.merged_df_dict[SPAMMER], self.merged_df_dict[LEGIT_USER]])
        self.df.sort_values(by=['UserID'], inplace=True)
        
        #Convert columns to datetime type
        datetime_cols = ['CreatedAt', 'CollectedAt']
        for col in datetime_cols:
            self.df[col] = pd.to_datetime(self.df[col])
                
        self.df_tweets = pd.concat([self.df_dict[f'{LEGIT_USER}_tweets'], self.df_dict[f'{SPAMMER}_tweets']])
        self.df_tweets['CreatedAt'] = pd.to_datetime(self.df_tweets['CreatedAt'])

    def to_csv(self):
        self.df.to_csv(f'./Cleaned_Data/features_raw.csv')

    def display_complete_feature(self):
        print(f'Feature {self.i} completed')
        self.i+=1
        
    def create_features(self):
        """Function to extract all the necessary features
        """
        list_features = ['UserID', 'SpammerBoolean']

        #1e feature: the length of the screen name
        list_features.append('LengthOfScreenName')
        self.display_complete_feature()

        #2e feature: the length of description
        list_features.append('LengthOfDescriptionInUserProfile')
        self.display_complete_feature()

        #3e feature: the longevity of the account
        self.df['AccountLongevity'] = (self.df['CollectedAt']-self.df['CreatedAt']).dt.days        
        list_features.append('AccountLongevity')
        self.display_complete_feature()

        
        #4e feature: Following's number
        list_features.append('NumberOfFollowings')
        self.display_complete_feature()
        
        #5e feature: Followers' number
        list_features.append('NumberOfFollowers')
        self.display_complete_feature()

        #6e feature: Standard deviation of unique series' following
        #self.df['StdDevSeriesUniqueFollowing'] = self.df['SeriesOfNumberofFollowings'].apply(lambda x: np.std(set(x.split(','))))
        self.df['StdDevSeriesUniqueFollowing'] = [list(set(map(int, i.strip('[]').split(',')))) for i in self.df['SeriesOfNumberofFollowings']]
        self.df['StdDevSeriesUniqueFollowing'] = self.df['StdDevSeriesUniqueFollowing'].apply(np.std)
        list_features.append('StdDevSeriesUniqueFollowing')
        self.display_complete_feature()
        
        #7e feature: Following on Followers ratio
        self.df['FonF_ratio'] = self.df['NumberOfFollowings'] / self.df['NumberOfFollowers']        
        list_features.append('FonF_ratio')
        self.display_complete_feature()

        #8e feature: Nombre de tweet
        list_features.append('NumberOfTweets')
        self.display_complete_feature()

        #9e feature: Total tweets sent per day
        self.df_tweets['CreatedAt_Days'] = self.df_tweets['CreatedAt'].values.astype('<M8[D]')
        #unique_date_tweet = self.df_tweets.groupby('UserID')['CreatedAt'].nunique().reset_index()
        tweet_per_day = self.df_tweets.groupby('UserID').size() / self.df_tweets.groupby('UserID')['CreatedAt_Days'].nunique()
        tweet_per_day = tweet_per_day.reset_index()
        tweet_per_day.columns = ['UserID', 'TweetSentPerDay']
        self.df = pd.merge(self.df, tweet_per_day, on='UserID', how='left')
        list_features.append('TweetSentPerDay')
        self.display_complete_feature()
        
        #10e feature: number of tweet on lifetime duration of account
        self.df['AvgTweetSentWhileActive'] = self.df['NumberOfTweets']/self.df['AccountLongevity']
        list_features.append('AvgTweetSentWhileActive')
        self.display_complete_feature()

        #11e feature: number of URL per tweet
        #self.df_tweets['URLcount'] = self.df_tweets['Tweet'].str.count('http')
        self.df_tweets['TweetURL'] = self.df_tweets['Tweet'].str.count('http')
        #tweet_url_total = self.df_tweets.groupby('UserID').agg({'TweetURL': 'sum'}).reset_index()
        tweet_url_total = self.df_tweets.loc[self.df_tweets['TweetURL']>0].groupby('UserID').count().reset_index()          
        self.df = pd.merge(self.df, tweet_url_total, on='UserID', how='left')
        
        nb_tweet = self.df_tweets.groupby('UserID')['Tweet'].size().reset_index()
        nb_tweet.columns = ['UserID', 'TweetSample']
        self.df = pd.merge(self.df, nb_tweet, on='UserID', how='left')
        #self.df['UrlOnTweet'] = self.df_tweets['URLcount'] / self.df_tweets.groupby('UserID')['CreatedAt'].nunique()
        #self.df['TweetUrlOnTweet'] = self.df['TweetURL'] / self.df['NumberOfTweets']
        self.df['TweetUrlOnTweet'] = self.df['TweetURL'] / self.df['TweetSample']

        list_features.append('TweetUrlOnTweet')
        self.display_complete_feature()

        #12e feature: average URL link per tweet sent
        #self.df_tweets['URLcount'] = self.df_tweets['Tweet'].str.count('http')
        url_total_per_users = self.df_tweets.groupby('UserID').agg({'TweetURL': 'sum'}).reset_index()
        url_total_per_users.columns = ['UserID', 'TotalURL']
        self.df = pd.merge(self.df, url_total_per_users, on='UserID', how='left')
        #self.df['AvgURLPerTweet'] = self.df['URLcount']/self.df['TweetSample']
        self.df['AvgURLPerTweet'] = self.df['TotalURL']/self.df['TweetSample']
        list_features.append('AvgURLPerTweet')
        self.display_complete_feature()

        #13e feature: mentions per tweet
        self.df_tweets['Mentions'] = self.df_tweets['Tweet'].str.count('([@])\w+')
        mentions_total = self.df_tweets.groupby('UserID').agg({'Mentions': 'sum'}).reset_index()
        self.df = pd.merge(self.df, mentions_total, on='UserID', how='left')
        self.df['MentionsPerTweet'] = self.df['Mentions']/self.df['TweetSample']
        list_features.append('MentionsPerTweet')
        self.display_complete_feature()

        #14e feature: avg time between tweet
        time_btw_tweet = (self.df_tweets.sort_values(['UserID','CreatedAt']).groupby('UserID')['CreatedAt'].diff().dt.total_seconds()/60).reset_index()
        time_btw_tweet.columns = ['UserID', 'Diff']
        avg_time_btw_tweet = time_btw_tweet.groupby('UserID')['Diff'].mean().reset_index()
        avg_time_btw_tweet.columns = ['UserID', 'AvgTimeBetween']
        self.df = pd.merge(self.df, avg_time_btw_tweet, on='UserID', how='left')
        list_features.append('AvgTimeBetween')
        self.display_complete_feature()

        #15e feature: max time between tweet
        max_time_btw_tw = time_btw_tweet.groupby('UserID')['Diff'].max().reset_index()
        max_time_btw_tw.columns = ['UserID', 'MaxTimeBetween']
        self.df = pd.merge(self.df, max_time_btw_tw, on='UserID', how='left')
        list_features.append('MaxTimeBetween')
        self.display_complete_feature()

        self.df = self.df[list_features]
        print(self.df.head())
        print(self.df.dtypes)
        print(self.df.isna().sum())  

# ===============================
# Order of scripts is as followed
# ===============================

#Create Features extracter class
df_obj = FeaturesExtracter()
#Create the filenames dictionary to automate the process
df_obj.create_filenames()
#Read all raw data into dataframe
df_obj.read_raw_csv()
#Remove all doubled rows
df_obj.verify_redundant_users()
#Create individual column to mark if user is spammer or not
df_obj.mark_spammer()
#Construct the main dataframe for the raw data
df_obj.create_main_dfs()
#Extract all 15 features from the raw data
df_obj.create_features()
#Save as csv file
df_obj.to_csv()



