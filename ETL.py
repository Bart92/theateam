import pandas as pd
import numpy as np
import data_helpers_neutrals

def parseTweets(fpath, fraction):
    labelColumn = 1
    textColumn = 2
    file = pd.read_csv(fpath, sep='\t', header=None)

    file_pos = open('data/rt-polarity.pos', "w")
    file_neg = open('data/rt-polarity.neg', "w")
    length = int(len(file)*fraction)
    numberOfMilestrones = 10 # how many times do you want to be informed of progress? 1=only final state reported, 2=halfway & end
    milestone = int(length/numberOfMilestrones)
    exceptionCatcher = []

    for i in range(0,length):
        if file[labelColumn][i]==1:
            file_pos.write(file[textColumn][i]+"\n")
        elif file[labelColumn][i]==0:
            file_neg.write(file[textColumn][i]+"\n")
        else:
            exceptionCatcher.append([i])
        if i%milestone==0:
            print('Parsed '+str(i)+r'/'+str(length)+' tweets with '+str(len(exceptionCatcher))+' exceptions')

    file_pos.close()
    file_neg.close()
    print('Job finished. Go home and drink a beer.')
    
def load_data():
    print("Load data...")
    x, y, vocabulary, vocabulary_inv_list, z = data_helpers_neutrals.load_data()
    print(x[1])
    print(z[1])
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    y = y.argmax(axis=1)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.9)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]
    print("... data loading complete.")
    return x_train, y_train, x_test, y_test, vocabulary_inv, z

def main(fpath, howMuchToParse):
    parseTweets(fpath, howMuchToParse)
    x_train, y_train, x_test, y_test, vocabulary_inv, neutral_tweets = load_data()
    return x_train, y_train, x_test, y_test, vocabulary_inv, neutral_tweets