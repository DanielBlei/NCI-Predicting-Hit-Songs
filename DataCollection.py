#-----------------------------------------------------------------------------------------------------------------------
# Project by Daniel Blei			Student ID: 16151704
#-----------------------------------------------------------------------------------------------------------------------
import spotipy
import spotipy.oauth2 as oauth2
import json
import pandas as pd
import itertools

#-----------------------------------------------------------------------------------------------------------------------
# Gather the Data
#-----------------------------------------------------------------------------------------------------------------------
#setting the ID and Secret key.

id='xxxxxxxxxx'
secret='xxxxxxxxxxx'
ruri='http://www.google.com'
username = ''

# --------------------------------------------------------------------------------------------------------------------
#Accessing the Spotify Web API, code supported by:
#    Title: Spotipy Refreshing a token with authorization code flow
#    Author: Henderson, N.
#    Date: 14th March 2018
#    Availability: https://stackoverflow.com/questions/49239516/spotipy-refreshing-a-token-with-authorization-code-flow
# --------------------------------------------------------------------------------------------------------------------
sp_oauth = oauth2.SpotifyOAuth(client_id=id,client_secret=secret,redirect_uri=ruri,scope='playlist-read-private')
token_info = sp_oauth.get_cached_token()
if not token_info:
    auth_url = sp_oauth.get_authorize_url()
    print(auth_url)
    response = input('Paste the above link into your browser, then paste the redirect url here: ')

    code = sp_oauth.parse_response_code(response)
    token_info = sp_oauth.get_access_token(code)

    token = token_info['access_token']

sp = spotipy.Spotify(auth=token)

#Function to refresh the token

def refresh():
    global token_info, sp
    token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
    token = token_info['access_token']
    sp = spotipy.Spotify(auth=token)
# --------------------------------------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------------------------------------
# Importing Lists of Artists ID
# --------------------------------------------------------------------------------------------------------------------
Artists_top = pd.read_csv('topArtistsID.csv') # 200 top artists
Artists_bot = pd.read_csv('BotArtistsID.csv') # 200 other artists

Songs_ID_Top=[]
Songs_ID_Bot=[]

#Getting the 10 most played tracks from the 200 most played artists:

i = 0
for i in range(len(Artists_top.ID)):
    results = sp.artist_top_tracks(Artists_top.ID[i])
    i = i + 1
    for track in results['tracks'][:10]:
        Songs_ID_Top.append(track['id'])
        #print 'Track  id: ' + track['id']
print 'Top Tracks Achieved'
TopArtistsTracks = pd.DataFrame(Songs_ID_Top)
TopArtistsTracks.to_csv('TopArtistsTracks.csv')

#Getting the 10 most played tracks from the artists randomly selected:

i = 0
for i in range(len(Artists_bot.ID)):
    results = sp.artist_top_tracks(Artists_bot.ID[i],country='IE')
    i = i + 1
    for track in results['tracks'][:10]:
        Songs_ID_Bot.append(track['id'])
        #print 'Track  id: ' + track['id']
print 'Other Tracks Achieved'
BotArtistsTracks = pd.DataFrame(Songs_ID_Bot)
BotArtistsTracks.to_csv('BotArtistsTracks.csv')
#--------------------------------------------------------------------------------------------------------------------
#   Getting Song Tracks Features
#--------------------------------------------------------------------------------------------------------------------
tracks_top = pd.read_csv('TopArtistsTracks.csv')
tracks_bot = pd.read_csv('BotArtistsTracks.csv')

tracks_top.rename(columns={'0': 'ID'},inplace=True)
tracks_bot.rename(columns={'0': 'ID'},inplace=True)

train_set_top = tracks_top.ID
train_set_bot = tracks_bot.ID
test_set_top = []
test_set_bot = []

# Creating the Training Set and Testing Set:
i=1
for i in range(0, len(tracks_top.ID), 10):
    test_set_top.append(tracks_top.ID[i])
    del train_set_top[i]

for i in range(0, len(tracks_bot.ID), 10):
    test_set_bot.append(tracks_bot.ID[i])
    del train_set_bot[i]
#-----------------------------------------------------------------------------------------------------------------------
# Gathering the Audio Features
#-----------------------------------------------------------------------------------------------------------------------
Songs_Features_train_T = []
Songs_Features_train_B = []

def audio_top_train():
    a = 0
    b = 1
    for i in range(0, len(train_set_top)):
        Songs_Features_train_T.append(sp.audio_features(train_set_top[a:b]))
        #Features = sp.audio_features(train_set_top[a:b])
        #print json.dumps(Features, indent=1)
        a = a + 1
        b = b + 1
    print "Audio Features Achieved"

def audio_bot_train():
    a = 0
    b = 1
    for i in range(0, len(train_set_bot)):
        Songs_Features_train_B.append(sp.audio_features(train_set_bot[a:b]))
        #Features = sp.audio_features(train_set_bot[a:b])
        #print json.dumps(Features, indent=1)
        a = a + 1
        b = b + 1
    print "Audio Features Achieved"

#Converting json to dataframe

def normalize_top_train():
    Songs= list(itertools.chain.from_iterable(Songs_Features_train_T))
    for i in Songs:
        if i == None:
            Songs.remove(None)
    TA_Songs_F = pd.read_json(json.dumps(Songs))
    TA_Songs_F.to_csv('train_set_top_AF.csv')

def normalize_bot_train():
    Songs= list(itertools.chain.from_iterable(Songs_Features_train_B))
    for i in Songs:
        if i == None:
            Songs.remove(None)
    TA_Songs_F = pd.read_json(json.dumps(Songs))
    TA_Songs_F.to_csv('train_set_bot_AF.csv')


#Gathering the Testing Set.

Songs_Features_test_T = []
Songs_Features_test_B = []

def audio_top_test():
    a = 0
    b = 1
    for i in range(0, len(test_set_top)):
        Songs_Features_test_T.append(sp.audio_features(test_set_top[a:b]))
        #Features = sp.audio_features(test_set_top[a:b])
        #print json.dumps(Features, indent=1)
        a = a + 1
        b = b + 1
    print "Audio Features Achieved"

def audio_bot_test():
    a = 0
    b = 1
    for i in range(0, len(test_set_bot)):
        Songs_Features_test_B.append(sp.audio_features(test_set_bot[a:b]))
        #Features = sp.audio_features(test_set_bot[a:b])
        #print json.dumps(Features, indent=1)
        a = a + 1
        b = b + 1
    print "Audio Features Achieved"

#Converting json to dataframe

def normalize_top_test():
    Songs= list(itertools.chain.from_iterable(Songs_Features_test_T))
    for i in Songs:
        if i == None:
            Songs.remove(None)
    TA_Songs_F = pd.read_json(json.dumps(Songs))
    TA_Songs_F.to_csv('test_set_top_AF.csv')

def normalize_bot_test():
    Songs= list(itertools.chain.from_iterable(Songs_Features_test_B))
    for i in Songs:
        if i == None:
            Songs.remove(None)
    TA_Songs_F = pd.read_json(json.dumps(Songs))
    TA_Songs_F.to_csv('test_set_bot_AF.csv')


