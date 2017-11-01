import data_helpers_neutrals
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
"""
This module gives a prediction (betwen 0 and 1) for a set of input sentences and calculates accuracy on whole set.
INPUT: a keras model, a vocabulary dictionary and a file with tweets to test
OUTPUT: accuracy
"""
def checkTweet(sentence, vocabulary_inv, model, sequence_length):
    #print(sentence)
    vocabulary = dict((v, k) for k, v in vocabulary_inv.items())
    words = sentence.split()
    #print(words)
    x_query = np.array([vocabulary[word] for word in words])
    #print(x_query)
    x_query = sequence.pad_sequences([x_query], maxlen=sequence_length, padding="post", truncating="post")
    y = model.predict(x_query, batch_size=1)
    return y

def val2sen(values):
    labels=['Strongly negative','Negative','Weakly Negative','Neutral','Weakly Positive','Positive','Strongly Positive']
    for index in range(0,len(labels)):
        while len(labels[index])<20:
            labels[index]+=' '
    intervals=[[0,0.1],[0.1,0.25],[0.25,0.35],[0.35,0.65],[0.65,0.75],[0.75,0.90],[0.9,1]]
    sentiments=[]
    for value in values:
        for interval in intervals:
            if value>=interval[0] and value<=interval[1]:
                sentiments.append(labels[intervals.index(interval)])
    return sentiments



def test(fpath, vocabulary_inv, model, sequence_length):
	print('Reading .tsv with tweets to test on...')
	sam_test_data = pd.read_csv(fpath,delimiter="\t",header=None,error_bad_lines=False)
	print('... tweets loaded.')
	positiveInstances = 0
	negativeInstances = 0
	confusionMatrix = [[0,0],[0,0]]
	numberOfSentences = len(sam_test_data[2])
	for i in range(0,numberOfSentences):
		pitstop = int(numberOfSentences/10)
		if i%pitstop==0:
			print(str(i)+'/'+str(len(sam_test_data[2])))
		tweet = sam_test_data[2][i]
		prediction = checkTweet(data_helpers_neutrals.clean_str(tweet), vocabulary_inv, model, sequence_length)
		if prediction>0.5:
			prediction = 1
		else:
			prediction = 0
		label = sam_test_data[1][i]
		if label==1:
			positiveInstances+=1
		elif label==0:
			negativeInstances+=1
		# build the Confusion Matrix
		if label==1:
			if prediction==label:
				confusionMatrix[0][0]+=1
			else:
				confusionMatrix[0][1]+=1
		if label==0:
			if prediction==label:
				confusionMatrix[1][0]+=1
			else:
				confusionMatrix[1][1]+=1
	# compute accuracy
	accuracy = (confusionMatrix[0][0]+confusionMatrix[1][1])/numberOfSentences
	print('PositiveInstances '+str(positiveInstances)+' NegativeInstances '+str(negativeInstances))
	print(confusionMatrix)
	print('Accuracy: '+str(accuracy))
	return confusionMatrix