#Fichier de pre-traitement 
#Auteur: Bui Trung Tin Michel BUIT31059004

#Chargement de librairie
import pandas as pd
import pickle as pck
import numpy as np

#Constant
SPAMMER = 'content_polluters'
LEGIT_USER = 'legitimate_users'

class DataFrameCreator():
    
    
    #Liste contenant nom des colonnes

    
    def __init__(self) -> None:
        #Importation des données en CSV
        self.df_dict = {}
        self.merged_df_dict = {}        

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

            print(f'Valeurs manquantes:{self.df_dict[file].isna().sum()}')
            print(f'Valeurs=\n{self.df_dict[file][self.df_dict[file].isna().any(axis=1)]}')
            print(f'Size of dataframe before drop duplicate={self.df_dict[file].shape}')
            self.df_dict[file].drop_duplicates(inplace=True)

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
        print(f'Size of dataframe after drop duplicate={self.df.shape}')
        self.df_tweets = pd.concat([self.df_dict[f'{LEGIT_USER}_tweets'], self.df_dict[f'{SPAMMER}_tweets']])

    def to_csv(self):
        self.df_tweets.to_csv(f'./Cleaned_Data/tweets.csv')
        self.df.to_csv(f'./Cleaned_Data/complete_users.csv')

    def load_data(self):
        self.df_users = pd.read_csv(f'./Cleaned_Data/complete_users.csv')
        self.df_tweets = pd.read_csv(f'./Cleaned_Data/tweets.csv')

        datetime_cols = ['CreatedAt', 'CollectedAt']
        for col in datetime_cols:
            self.df_users[col] = pd.to_datetime(self.df_users[col])
        self.df_tweets['CreatedAt'] = pd.to_datetime(self.df_tweets['CreatedAt'])

        self.df = self.df_users.merge(self.df_tweets, on='UserID', how='right').groupby('UserID')

    #1e feature: the length of the screen name
    def length_screen_name(self):
        print(self.df_users['LengthOfScreenName'])

    #2e feature: the length of description
    def length_description_msg(self):
        pass
    
    #3e feature: the longevity of the account
    def longevity_account(self):
        #self.df_users['Longevity'] = (self.df_tweets.groupby(by=['UserID'])['CreatedAt'].max() - self.df_tweets.groupby(by=['UserID'])['CreatedAt'].min()).dt.days
        self.df_users['Longevity'] = (self.df_users['CollectedAt']-self.df_users['CreatedAt']).dt.days
      
        #print(longevity_series.dtypes)
        #return longevity_series

    #4e feature
    def following_number(self):
        #print(50*'=')
        #print(self.df_users['NumberOfFollowings'])
        return self.df_users['NumberOfFollowings']
    
    #5e feature
    def followers_number(self):
        #print(50*'=')
        #print(self.df_users['NumberOfFollowers'])
        return self.df_users['NumberOfFollowers']
    
    
    #6e feature
    def std_num_unique_following(self):
        print(50*'=')
        print(self.df_users['SeriesOfNumberofFollowings'])
        print(50*'=')
        print(self.df_users['SeriesOfNumberofFollowings'].apply(lambda x: len(set(x.split(',')))))
        

    #7e feature
    def following_on_followers_ratio(self):
        #TODO Also outputs INF
        ratio = self.df_users['NumberOfFollowings'] / self.df_users['NumberOfFollowers']        
        print(ratio)
    
    #8e feature
    def total_tweet_sent(self):
        print(self.df_users['NumberOfTweets'])
    
    #9e feature
    def total_tweet_sent_perday(self):
        #self.df_tweets.dat.apply(lambda dt: dt.date()).groupby([self.df_tweets['UserID'], self.df_tweets['CreatedAt'].dat.apply(lambda dt: dt.year)]).nunique()
        #TODO wait until teacher replies
        self.df_tweets['CreatedAt'] = self.df_tweets['CreatedAt'].values.astype('<M8[D]')
        print(self.df_tweets.groupby('UserID')['CreatedAt'].nunique())
        print(self.df_tweets.groupby('UserID').size())

    
    #10e feature
    def total_tweet_on_account_lifetime_ratio(self):
        #TODO Also outputs INF

        self.df_users['TotalTweetSentPerDay'] = self.df_users['NumberOfTweets']/self.df_users['Longevity']
    
    #11e feature
    def URL_per_tweet_ratio(self):
        #print(self.df_tweets.groupby('UserID')['Tweet'].apply(lambda x: x[x.str.contains('http')].count()))
        #print(self.df_tweets['Tweet'].str.count("http"))
        #print(self.df_tweets.groupby('UserID').size())
        #print(self.df_tweets['Tweet'].str.count("http")/self.df_tweets.groupby('UserID').size())
        self.df_tweets['URLcount'] = self.df_tweets['Tweet'].str.count('http')
        
        url_total = self.df_tweets.groupby('UserID').agg({'URLcount': 'sum'})
        url_total = url_total.reset_index()
        print(url_total)
        nb_tweet = self.df_tweets.groupby('UserID')['Tweet'].size()
        nb_tweet = nb_tweet.reset_index()
        url_tweet = pd.merge(nb_tweet, url_total, on='UserID', how='left')
        url_tweet['URLperTweet'] = url_tweet['URLcount']/url_tweet['Tweet']
        print(url_tweet)
        
        
        #self.df_users['URLperTweet'] 
        #print(self.df_users['URLperTweet'])
        
    #12e feature
    def avg_URL_per_tweet(self):
        pass
    #13e feature
    def mentions_per_tweet_ratio(self):
        self.df_tweets['Mentions'] = self.df_tweets['Tweet'].str.count('([@])\w+')
        print(self.df_tweets['Mentions'])
    #14e feature
    def avg_time_between_tweet(self):
        #avg_time_btw_tweet = (self.df_tweets.groupby(by=['UserID'])['CreatedAt'].max() - self.df_tweets.groupby(by=['UserID'])['CreatedAt'].min()).dt.days
        self.df_tweets['Diff'] = self.df_tweets.sort_values(['UserID','CreatedAt']).groupby('UserID')['CreatedAt'].diff().dt.seconds
        #self.df_tweets.to_csv(f'./tweets.csv')
        avg_time_btw_tw = self.df_tweets.groupby('UserID')['Diff'].mean()
        avg_time_btw_tw = avg_time_btw_tw.reset_index()
        #print(avg_time_btw_tw.tail())
        avg_time_btw_tw.columns = ['UserID', 'AvgTimeBetween']
        self.df_users = pd.merge(self.df_users, avg_time_btw_tw, on='UserID', how='left')
        #self.df_users['Diff'].fillna(self.df_users['Diff'].mean(), inplace=True)
        #self.df_users['AvgTimeBetween'] = self.df_users['AvgTimeBetween'].fillna(self.df_users['AvgTimeBetween'].mean())
        #print(self.df_users.tail())
        #self.df_tweets.to_csv('./new_df.csv')
        print(self.df_tweets.sort_values(['UserID'], ascending=False).head())
    #15e feature
    def max_time_between_tweet(self):
        max_time_btw_tw = self.df_tweets.groupby('UserID')['Diff'].max().reset_index()
        max_time_btw_tw.columns = ['UserID', 'MaxTimeBetween']
        #print(max_time_btw_tw.head())
        self.df_users = pd.merge(self.df_users, max_time_btw_tw, on='UserID', how='left')
        #self.df_users['MaxTimeBetween'] = self.df_users['MaxTimeBetween'].fillna(self.df_users['MaxTimeBetween'].mean())
        #print(self.df_users.tail())
    #Optional feature



#Creation de dictionnaire contenant les noms des fichiers avec la
#liste des noms de colonnes

df_obj = DataFrameCreator()
df_obj.create_filenames()
df_obj.read_raw_csv()
df_obj.mark_spammer()
df_obj.create_main_dfs()
df_obj.to_csv()




#df.to_csv(f'./Cleaned_Data/complete_users.csv')
#df.to_pickle(f'./Cleaned_Data/complete_users.pkl')
#merged_tweets_df.to_pickle(f'./Cleaned_Data/tweets.pkl')

