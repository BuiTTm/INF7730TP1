#Features extraction 
#Auteur: Bui Trung Tin Michel BUIT31059004
from numpy import datetime64
import pandas as pd

class FeatureExtracter():
    
    def __init__(self) -> None:
        #self.df_users = pd.read_pickle(f'./Cleaned_Data/complete_users.pkl')
        self.df_users = pd.read_csv(f'./Cleaned_Data/complete_users.csv')
        #self.df_tweets = pd.read_pickle(f'./Cleaned_Data/tweets.pkl')
        self.df_tweets = pd.read_csv(f'./Cleaned_Data/tweets.csv')
        #print(self.df_users.dtypes)
        datetime_cols = ['CreatedAt', 'CollectedAt']
        for col in datetime_cols:
            self.df_users[col] = pd.to_datetime(self.df_users[col])
        self.df_tweets['CreatedAt'] = pd.to_datetime(self.df_tweets['CreatedAt'])
        #self.df_users.astype(col_type)
        self.df = self.df_users.merge(self.df_tweets, on='UserID', how='right').groupby('UserID')
        print(50*'*')
        print(self.df.head())
        #print(self.df_users.dtypes)
        #print(self.df_tweets.dtypes)


    #1e feature: the length of the screen name
    def length_screen_name(self):
        print(self.df_users['LengthOfScreenName'])

    #2e feature: the length of description
    def length_description_msg(self):
        pass
    
    #3e feature: the longevity of the account
    def longevity_account(self):
        #print(self.df_users['CreatedAt'])
        # print(50*'=')
        # print(self.df_tweets.groupby(by=['UserID'])['CreatedAt'].max())
        # print(50*'=')
        # print(self.df_tweets.groupby(by=['UserID'])['CreatedAt'].min())
        # print(50*'=')
        #self.df_users['Longevity'] = (self.df_tweets.groupby(by=['UserID'])['CreatedAt'].max() - self.df_tweets.groupby(by=['UserID'])['CreatedAt'].min()).dt.days
        self.df_users['Longevity'] = (self.df_users['CollectedAt']-self.df_users['CreatedAt']).dt.days

        print(50*'+')        
        print(self.df_users['Longevity'])
        print(50*'+')        
        #print(longevity_series.dtypes)

        #return longevity_series
        #pass   
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

        #print(50*'=')
        #print(self.df_users['NumberOfTweets'].dtype)
        #print(50*'=')
        #print(tweet_perday)
    
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
        pass
    #14e feature
    def avg_time_between_tweet(self):
        pass
    #15e feature
    def max_time_between_tweet(self):
        pass
    #Optional feature


features_extracter = FeatureExtracter()
#features_extracter.length_screen_name()
features_extracter.longevity_account()
#features_extracter.following_number()
#features_extracter.followers_number()
#features_extracter.std_num_unique_following()
#features_extracter.following_on_followers_ratio()
#features_extracter.total_tweet_sent_perday()
features_extracter.URL_per_tweet_ratio()
cols = ['LengthOfScreenName', 'LengthOfDescriptionInUserProfile']