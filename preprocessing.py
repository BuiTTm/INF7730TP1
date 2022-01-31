#Fichier de pre-traitement 
#Auteur: Bui Trung Tin Michel BUIT31059004

import pandas as pd
import pickle as pick

#Columns names depending on which type of data it is
user_cols = ['UserID', 'CreatedAt', 'CollectedAt', 
     'NumberOfFollowings', 'NumberOfFollowers', 
     'NumberOfTweets', 'LengthOfScreenName', 
     'LengthOfDescriptionInUserProfile']
follow_cols = ['UserID', 'SeriesOfNumberofFollowings']
tweet_cols = ['UserID', 'TweetID', 'Tweet', 'CreatedAt']

#Filename with each columns names
SPAMMER = 'content_polluters'
LEGIT_USER = 'legitimate_users'
files = {
    SPAMMER: user_cols, 
    f'{SPAMMER}_followings': follow_cols, 
    f'{SPAMMER}_tweets': tweet_cols,
    LEGIT_USER: user_cols, 
    f'{LEGIT_USER}_followings': follow_cols, 
    f'{LEGIT_USER}_tweets': tweet_cols
    }
df_dict = {}
merged_df_dict = {}

#Read text files into dataframes
for file, cols in files.items():
    df_dict[file] = pd.read_csv(f'./social_honeypot_icwsm_2011/{file}.txt', sep='\t', names=cols, index_col='UserID')
    print(df_dict[file].head())
 
#Create columns indicating if users is a spammer or not
for user in [SPAMMER, LEGIT_USER]:   
    merged_df_dict[user] = df_dict[user]\
        .merge(df_dict[f'{user}_followings'], on='UserID') 
    if user is SPAMMER:
        merged_df_dict[user]['SpammerBoolean']=1
    elif user is LEGIT_USER:
        merged_df_dict[user]['SpammerBoolean']=0
    print(merged_df_dict[user].head(10))
    print(f'Columns={merged_df_dict[user].columns}')
    merged_df_dict[user].to_csv(f'./Cleaned_Data/{user}.csv')

#Merge users into one dataframe
df = pd.concat([merged_df_dict[SPAMMER], merged_df_dict[LEGIT_USER]])
df.sort_values(by=['UserID'], inplace=True)
df.to_csv(f'./Cleaned_Data/complete_users.csv')
