#Fichier de pre-traitement 
#Auteur: Bui Trung Tin Michel BUIT31059004

#Chargement de librairie
import pandas as pd
import pickle as pck
import numpy as np
#import sklearn.preprocessing import Imputer
#Constant strings
SPAMMER = 'content_polluters'
LEGIT_USER = 'legitimate_users'

class DataFrameCreator():
    
    
    #Liste contenant nom des colonnes

    
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

            print(f'Valeurs manquantes:{self.df_dict[file].isna().sum()}')
            print(f'Valeurs=\n{self.df_dict[file][self.df_dict[file].isna().any(axis=1)]}')
            print(f'Size of dataframe before drop duplicate={self.df_dict[file].shape}')
            self.df_dict[file].drop_duplicates(inplace=True)
            print(f'Size of dataframe after drop duplicate={self.df_dict[file].shape}')

            
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
        #self.df_tweets.to_csv(f'./Cleaned_Data/tweets.csv')
        self.df.to_csv(f'./Cleaned_Data/dataset.csv')

    # def load_data(self):
    #     self.df_users = pd.read_csv(f'./Cleaned_Data/complete_users.csv')
    #     self.df_tweets = pd.read_csv(f'./Cleaned_Data/tweets.csv')

    #     datetime_cols = ['CreatedAt', 'CollectedAt']
    #     for col in datetime_cols:
    #         self.df_users[col] = pd.to_datetime(self.df_users[col])
    #     self.df_tweets['CreatedAt'] = pd.to_datetime(self.df_tweets['CreatedAt'])

    #     self.df = self.df_users.merge(self.df_tweets, on='UserID', how='right').groupby('UserID')

    def display_complete_feature(self):
        print(f'Feature {self.i} completed')
        self.i+=1
        
    def create_features(self):
        list_features = ['UserID', 'SpammerBoolean']

        #1e feature: the length of the screen name
        list_features.append('LengthOfScreenName')
        self.display_complete_feature()

        #2e feature: the length of description
        list_features.append('LengthOfDescriptionInUserProfile')
        self.display_complete_feature()

        #3e feature: the longevity of the account
        #last_active = self.df_tweets.groupby('UserID')['CreatedAt'].max().reset_index()
        #last_active.columns = ['UserID', 'LastActive']
        #self.df = pd.merge(self.df, last_active, on='UserID', how='left')
        #self.df['LastActive'] = pd.to_datetime(self.df['LastActive'])
        #self.df['AccountLongevity'] = (self.df['LastActive']-self.df['CreatedAt']).dt.days
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
        self.df['StdDevSeriesUniqueFollowing'] = [list(set(map(int, i.split(',')))) for i in self.df['SeriesOfNumberofFollowings']]
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
        #TODO Also outputs INF
        self.df['AvgTweetSentWhileActive'] = self.df['NumberOfTweets']/self.df['AccountLongevity']
        list_features.append('AvgTweetSentWhileActive')
        self.display_complete_feature()

        #11e feature: number of URL per tweet
        self.df_tweets['URLcount'] = self.df_tweets['Tweet'].str.count('http')
        url_total = self.df_tweets.groupby('UserID').agg({'URLcount': 'sum'}).reset_index()        
        self.df = pd.merge(self.df, url_total, on='UserID', how='left')
        #self.df['UrlOnTweet'] = self.df_tweets['URLcount'] / self.df_tweets.groupby('UserID')['CreatedAt'].nunique()
        self.df['UrlOnTweet'] = self.df['URLcount'] / self.df['NumberOfTweets']
        list_features.append('UrlOnTweet')
        self.display_complete_feature()

        #12e feature: average URL link per tweet sent
        nb_tweet = self.df_tweets.groupby('UserID')['Tweet'].size().reset_index()
        nb_tweet.columns = ['UserID', 'TweetSample']
        self.df = pd.merge(self.df, nb_tweet, on='UserID', how='left')
        self.df['AvgURLPerTweet'] = self.df['URLcount']/self.df['TweetSample']
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
        
          
    def fill_missing_data(self):
        pass
        
    # #1e feature: the length of the screen name
    # def length_screen_name(self):
    #     print(self.df_users['LengthOfScreenName'])

    # #2e feature: the length of description
    # def length_description_msg(self):
    #     pass
    
    # #3e feature: the longevity of the account
    # def longevity_account(self):
    #     #self.df_users['Longevity'] = (self.df_tweets.groupby(by=['UserID'])['CreatedAt'].max() - self.df_tweets.groupby(by=['UserID'])['CreatedAt'].min()).dt.days
    #     self.df_users['Longevity'] = (self.df_users['CollectedAt']-self.df_users['CreatedAt']).dt.days
      
    #     #print(longevity_series.dtypes)
    #     #return longevity_series

    # #4e feature
    # def following_number(self):
    #     #print(50*'=')
    #     #print(self.df_users['NumberOfFollowings'])
    #     return self.df_users['NumberOfFollowings']
    
    # #5e feature
    # def followers_number(self):
    #     #print(50*'=')
    #     #print(self.df_users['NumberOfFollowers'])
    #     return self.df_users['NumberOfFollowers']
    
    
    # #6e feature
    # def std_num_unique_following(self):
    #     print(50*'=')
    #     print(self.df_users['SeriesOfNumberofFollowings'])
    #     print(50*'=')
    #     print(self.df_users['SeriesOfNumberofFollowings'].apply(lambda x: len(set(x.split(',')))))
        

    # #7e feature
    # def following_on_followers_ratio(self):
    #     #TODO Also outputs INF
    #     ratio = self.df_users['NumberOfFollowings'] / self.df_users['NumberOfFollowers']        
    #     print(ratio)
    
    # #8e feature
    # def total_tweet_sent(self):
    #     print(self.df_users['NumberOfTweets'])
    
    # #9e feature
    # def total_tweet_sent_perday(self):
    #     #self.df_tweets.dat.apply(lambda dt: dt.date()).groupby([self.df_tweets['UserID'], self.df_tweets['CreatedAt'].dat.apply(lambda dt: dt.year)]).nunique()
    #     #TODO wait until teacher replies
    #     self.df_tweets['CreatedAt'] = self.df_tweets['CreatedAt'].values.astype('<M8[D]')
    #     print(self.df_tweets.groupby('UserID')['CreatedAt'].nunique())
    #     print(self.df_tweets.groupby('UserID').size())

    
    # #10e feature
    # def total_tweet_on_account_lifetime_ratio(self):
    #     #TODO Also outputs INF
    #     self.df_users['TotalTweetSentPerDay'] = self.df_users['NumberOfTweets']/self.df_users['Longevity']
    
    # #11e feature
    # def URL_per_tweet_ratio(self):
    #     #print(self.df_tweets.groupby('UserID')['Tweet'].apply(lambda x: x[x.str.contains('http')].count()))
    #     #print(self.df_tweets['Tweet'].str.count("http"))
    #     #print(self.df_tweets.groupby('UserID').size())
    #     #print(self.df_tweets['Tweet'].str.count("http")/self.df_tweets.groupby('UserID').size())
    #     self.df_tweets['URLcount'] = self.df_tweets['Tweet'].str.count('http')
        
    #     url_total = self.df_tweets.groupby('UserID').agg({'URLcount': 'sum'})
    #     url_total = url_total.reset_index()
    #     print(url_total)
    #     nb_tweet = self.df_tweets.groupby('UserID')['Tweet'].size()
    #     nb_tweet = nb_tweet.reset_index()
    #     url_tweet = pd.merge(nb_tweet, url_total, on='UserID', how='left')
    #     url_tweet['URLperTweet'] = url_tweet['URLcount']/url_tweet['Tweet']
    #     print(url_tweet)
        
        
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
        self.df_tweets['Diff'] = self.df_tweets.sort_values(['UserID','CreatedAt'], ascending=False).groupby('UserID')['CreatedAt'].diff().dt.minutes
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
df_obj.verify_redundant_users()
df_obj.mark_spammer()
df_obj.create_main_dfs()
df_obj.create_features()
df_obj.to_csv()





#df.to_csv(f'./Cleaned_Data/complete_users.csv')
#df.to_pickle(f'./Cleaned_Data/complete_users.pkl')
#merged_tweets_df.to_pickle(f'./Cleaned_Data/tweets.pkl')

