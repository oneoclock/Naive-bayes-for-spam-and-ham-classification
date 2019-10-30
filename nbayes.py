
import re
import random
import math
import matplotlib.pyplot as plt

random.seed(10)
"""
Read text data from file and pre-process text by doing the following
1. convert to lowercase
2. convert tabs to spaces
3. remove "non-word" characters
Store resulting "words" into an array
"""
# Change FILENAME variable to the path pointing to the file SMSSpamCOllection.
FILENAME='C:/Users/Hiral/Desktop/IML/SMSSpamCollection'
all_data = open(FILENAME).readlines()

# randomizing and splitting into train and test
num_samples = len(all_data)
all_idx = list(range(num_samples))
random.shuffle(all_idx)
idx_limit = int(0.8*num_samples)
train_idx = all_idx[:idx_limit]
test_idx = all_idx[idx_limit:]
train_examples = [all_data[ii] for ii in train_idx]
test_examples = [all_data[ii] for ii in test_idx]

# Preprocess train and test examples
train_words = []
train_labels = []
test_words = []
test_labels = []

# train examples
for line in train_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige returne
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to spae
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    label = 1 if label == 'spam' else 0
    line_words = line_words[1:]
    train_words.append(line_words)
    train_labels.append(label)

# test examples
for line in test_examples:
    line = line.strip('\r\n\t ')  # remove trailing spaces, tabs and carraige return
    line = line.lower()  # lowercase
    line = line.replace("\t", ' ')  # convert tabs to spae
    line_words = re.findall(r'\w+', line)
    line_words = [xx for xx in line_words if xx != '']  # remove empty words

    label = line_words[0]
    label = 1 if label == 'spam' else 0
    line_words = line_words[1:]
    test_words.append(line_words)
    test_labels.append(label)
'''
train_words, test_words are list of lists
'''
spam_words = []
ham_words = []
alpha = 0.03125 #(1/32)
alpha2 = 0.0625 #1/16
alpha3 = 0.125  #1/8
alpha4 = 0.25   #1/4
alpha5 = 0.5    #1/2
alpha6 = 1      #1/1

#-----collecting all words used in spam in one variable spam_words

for ii in range(len(train_words)):  # we pass through words in each (train) SMS
    words = train_words[ii]
    label = train_labels[ii]
    if label == 1:
        spam_words += words
    else:
        ham_words += words
input_words = spam_words + ham_words  # all words in the input vocabulary

# Count spam and ham occurances for each word

#different spam and ham counts due to different alpha values
spam_counts = {};  ham_counts = {}
spam_counts2 = {}; ham_counts2 = {}
spam_counts3 = {}; ham_counts3 = {}
spam_counts4 = {}; ham_counts4 = {}
spam_counts5 = {}; ham_counts5 = {}
spam_counts6 = {}; ham_counts6 = {}
#dictionary of spam words with the no. of times they occur

for word in spam_words:
    try:
        word_spam_count = spam_counts.get(word)
        spam_counts[word] = word_spam_count + 1
        spam_counts2[word] = word_spam_count + 1
        spam_counts3[word] = word_spam_count + 1
        spam_counts4[word] = word_spam_count + 1
        spam_counts5[word] = word_spam_count + 1
        spam_counts6[word] = word_spam_count + 1
    except:
        spam_counts[word] = 1 + alpha  # smoothening
        spam_counts2[word] = 1 + alpha2
        spam_counts3[word] = 1 + alpha3
        spam_counts4[word] = 1 + alpha4
        spam_counts5[word] = 1 + alpha5
        spam_counts6[word] = 1 + alpha6

#dictionary of spam words with the no. of times they occur
for word in ham_words:
    try:
        word_ham_count = ham_counts.get(word)
        ham_counts[word] = word_ham_count + 1
        ham_counts2[word] = word_ham_count + 1
        ham_counts3[word] = word_ham_count + 1
        ham_counts4[word] = word_ham_count + 1
        ham_counts5[word] = word_ham_count + 1
        ham_counts6[word] = word_ham_count + 1
    except:
        ham_counts[word] = 1 + alpha  # smoothening
        ham_counts2[word] = 1 + alpha2  # smoothening
        ham_counts3[word] = 1 + alpha3  # smoothening
        ham_counts4[word] = 1 + alpha4  # smoothening
        ham_counts5[word] = 1 + alpha5  # smoothening
        ham_counts6[word] = 1 + alpha6  # smoothening  
'''
ham_words is list of all words that occur in ham msgs with repeatation
spam_words is list of all words that occur in spam msgs with repeatation

spam_counts a dictionary where the key is the word in spam and value is the no of occurrence
ham_counts
'''
num_spam = len(spam_words)
num_ham = len(ham_words)

predicted_label1=[]; predicted_label2=[]; predicted_label3=[]; predicted_label4=[]
predicted_label5=[]; predicted_label6=[]


'''
The predict function checks probability of words being in spam and ham.
And then stores 1 (spam) or 0 (ham) in predicted_label list whichever's probability high.
'''
def predict(alphax,spam_countss,ham_countss,data):
    prob_spam=0; prob_ham=0
    predicted_label=[]
    for i in range(len(data)):
        prob_spam=0; prob_ham=0
        for j in data[i]:
             try:
                    prob_spam+=math.log(spam_countss[j] / (num_spam + alphax * 20000))
             except:
                    prob_spam+=math.log(0 + alphax / (num_spam + alphax * 20000))

             try:
                    prob_ham+=math.log(ham_countss[j] / (num_ham + alphax * 20000))
             except:
                    prob_ham+=math.log(0 + alphax / (num_ham + alphax * 20000))
        if prob_spam>prob_ham:
            predicted_label.append(1)
        else:
            predicted_label.append(0)

    return predicted_label

'''
The scores function outputs the F-Score Accuracy, True_positive etc. for a model when called.
'''
def scores(predicted_label,labels):
    count=0
    true_positive=0
    false_positive=0
    false_negative=0
    true_negative=0
    for i in range(len(labels)):
        if labels[i]==predicted_label[i]:
            if predicted_label[i]==1:
                true_positive+=1
                count+=1
            elif predicted_label[i]==0:
                true_negative+=1
                count+=1
        elif predicted_label[i] == 1:
            false_positive+=1
        elif predicted_label[i] == 0:
            false_negative+=1
    
    print("Total data(no. of smses: )",len(labels))
    print("Correctly predicted:",count)
    accuracy=count/(len(labels))
    print("Accuracy:",accuracy)
    print("True_Positive\tFalse_Positive\tFalse_Negative\tTrue_Negative")
    print(true_positive,"\t\t",false_positive,"\t\t",false_negative,"\t\t",true_negative)

    precision=true_positive/(true_positive+false_positive)
    recall=true_positive/(true_positive+false_negative)
    fScore=2*((precision*recall)/(precision+recall))

    print("Precision:",precision)
    print("Recall:",recall)
    print("F_Score:",fScore,"\n\n")
    return fScore, accuracy

predicted_label1=predict(alpha,spam_counts,ham_counts,test_words)
predicted_label2=predict(alpha2,spam_counts2,ham_counts2,test_words)
predicted_label3=predict(alpha3,spam_counts3,ham_counts3,test_words)
predicted_label4=predict(alpha4,spam_counts4,ham_counts4,test_words)
predicted_label5=predict(alpha5,spam_counts5,ham_counts5,test_words)
predicted_label6=predict(alpha6,spam_counts6,ham_counts6,test_words)

print("Scores for Test Set with i=5")
fscore1, acc1=scores(predicted_label1,test_labels)
print("Scores for Test Set with i=4")
fscore2, acc2=scores(predicted_label2,test_labels)
print("Scores for Test Set with i=3")
fscore3, acc3=scores(predicted_label3,test_labels)
print("Scores for Test Set with i=2")
fscore4, acc4=scores(predicted_label4,test_labels)
print("Scores for Test Set with i=1")
fscore5, acc5=scores(predicted_label5,test_labels)
print("Scores for Test Set with i=0")
fscore6, acc6=scores(predicted_label6,test_labels)


#Graph plot for test set
i_alpha=[-5,-4,-3,-2,-1,0]
fs=[fscore1,fscore2,fscore3,fscore4,fscore5,fscore6]
accs=[acc1,acc2,acc3,acc4,acc5,acc6]
plt.plot(i_alpha,fs,color="blue",label="F_Score")
plt.plot(i_alpha,accs,color="red",label="Accuracy")
plt.xlabel('i Values')
plt.ylabel('Accuracy and F Scores')
plt.legend()
plt.show()

#Calling predict function for different aplha values.
predicted_label1=predict(alpha,spam_counts,ham_counts,train_words)
predicted_label2=predict(alpha2,spam_counts2,ham_counts2,train_words)
predicted_label3=predict(alpha3,spam_counts3,ham_counts3,train_words)
predicted_label4=predict(alpha4,spam_counts4,ham_counts4,train_words)
predicted_label5=predict(alpha5,spam_counts5,ham_counts5,train_words)
predicted_label6=predict(alpha6,spam_counts6,ham_counts6,train_words)

#Printing scores for each model.
print("Scores for Training Set with i=5")
fscore1, acc1=scores(predicted_label1,train_labels)
print("Scores for Training Set with i=4")
fscore2, acc2=scores(predicted_label2,train_labels)
print("Scores for Training Set with i=3")
fscore3, acc3=scores(predicted_label3,train_labels)
print("Scores for Training Set with i=2")
fscore4, acc4=scores(predicted_label4,train_labels)
print("Scores for Training Set with i=1")
fscore5, acc5=scores(predicted_label5,train_labels)
print("Scores for Training Set with i=0")
fscore6, acc6=scores(predicted_label6,train_labels)


#Graph plot for train set
i_alpha=[-5,-4,-3,-2,-1,0]
fs=[fscore1,fscore2,fscore3,fscore4,fscore5,fscore6]
accs=[acc1,acc2,acc3,acc4,acc5,acc6]
plt.plot(i_alpha,fs,color="blue",label="F_Score")
plt.plot(i_alpha,accs,color="red",label="Accuracy")
plt.xlabel('i Values')
plt.ylabel('Accuracy and F Scores')
plt.legend()
plt.show()
