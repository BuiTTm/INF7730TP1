#Features extraction 
#Auteur: Bui Trung Tin Michel BUIT31059004
import pandas as pd

class FeatureExtracter():
    
    def __init__(self) -> None:
        self.df_users = pd.read_pickle(f'./Cleaned_Data/complete_users.pkl')
        self.df_tweets = pd.read_pickle(f'./Cleaned_Data/tweets.pkl')
        print(self.df_users.columns)
        print(self.df_users.head(5))
        print(self.df_tweets.columns)
        print(self.df_tweets.head(5))

    #1e feature: the length of the screen name
    def length_screen_name(self):
        print(self.df_users['LengthOfScreenName'])

    #2e feature: the length of description
    def length_description_msg(self):
        pass
    
    #3e feature: the longevity of the account
    def longevity_account(self):
        pass
    
    #4e feature
    def following_number(self):
        pass
    
    #5e feature
    def followers_number(self):
        pass
    
    #6e feature
    def std_num_unique_following(self):
        pass
    
    #7e feature
    def following_on_followers_ratio(self):
        pass
    
    #8e feature
    def total_tweet_sent(self):
        pass
    
    #9e feature
    def total_tweet_sent_perday(self):
        pass
    
    #10e feature
    def total_tweet_on_account_lifetime_ratio(self):
        pass
    
    #11e feature
    def URL_per_tweet_ratio(self):
        pass
    
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
features_extracter.length_screen_name()