import data_helpers_neutrals
import pandas as pd
import numpy as np
from keras.preprocessing import sequence

def checkTweet(sentence, vocabulary, model, sequence_length):
    #print(sentence)
    words = sentence.split()
    #print(words)
    x_query = np.array([vocabulary[word] for word in words])
    #print(x_query)
    x_query = sequence.pad_sequences([x_query], maxlen=sequence_length, padding="post", truncating="post")
    y = model.predict(x_query, batch_size=1)
    return y


def test(fpath, vocabulary_inv, model, sequence_length):
    
    import pandas as pd
    
    companies = ["mylan","telia","volkswagen","samsung"]
    for company in companies:
        tweets = pd.read_csv(r'/Users/Bart/Desktop/AITeam/TweetsSplitByCompany/'+company+'Tweets.txt',delimiter="\t",error_bad_lines=False,header=None,index_col=None)
        output = open(r'/Users/Bart/Desktop/AITeam/TweetsSplitByCompany/'+company+'Guesses.txt','a')

        vocabulary = dict((v, k) for k, v in vocabulary_inv.items())

        for i in range(len(tweets)):
            original = str(tweets.iloc[i,0])+"\t"+str(tweets.iloc[i,1])
            tweet = tweets.iloc[i,1]
            prediction = checkTweet(data_helpers_neutrals.clean_str(tweet), vocabulary, model, sequence_length)
            if prediction>0.5:
                prediction = 1
            else:
                prediction = 0
            output.write(str(prediction)+'\t'+str(original)+'\n')

        output.close()
