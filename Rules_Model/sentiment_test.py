import nltk
import ssl
import numpy as np
import pandas as pd
import difflib as dfl
import spacy
nlp = spacy.load("en_core_web_sm")

#only need to run first time
#try:
#    _create_unverified_https_category = ssl._create_unverified_category
#except AttributeError:
#    pass
#else:
#    ssl._create_default_https_category = _create_unverified_https_category

#nltk.download("vader_lexicon") # downloads vader_lexicon to /Users/nbachman/nltk_data
#print("import nltk,vader and SentimentIntesity complete")

#Domain specific Heuristics
#Create a Lexicon to determine if Democrat or Republican words are the topic (Theta)
R_Lexicon = ["republican","republicans","republican's","gop", "g.o.p","conservative","conservatives","trump","trump's","pence","ryan","mcconnell","bush","mccarthy", "scalise", "boehner","romney","cruz","rubio","cain","giuliani"]
D_Lexicon = ["democrat","democrats","democrat's","democratic","liberal","liberals","clinton","clinton's","biden","biden's","pelosi","schumer","reid","hoyer","clyburn","obama","harris","kaine","sanders","feinstein"]

#Import Sentiment analyzer from VADER 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Update the VADER Lexicon to contorl how much Positive or Negative Weight.  This can be tuned. 
NEG = -3
POS = 3
NET = 0

#These Lexicons put more weight on popular politcaly charged terms.   They are created using domain knowledge and not probabilities
NEG_Lexicon = {u'losers': NEG, u'lose': NEG,u'impeached': NEG,u'landslide': NEG,u'repeal': NEG,u'replace': NEG,u'delusional': NEG,u'hackers': NEG,u'hate': NEG,u'terror': NEG,u'threat': NEG,u'kill': NEG,u'fear': NEG,u'destruction': NEG,u'danger': NEG, u'murder': NEG, u'enemies': NEG, u'hard': NEG, u'war': NEG, u'confront': NEG, u'threats': NEG, u'terrorism': NEG, u'problems': NEG, u'difficult': NEG, u'conflict': NEG, u'tyranny': NEG, u'evil': NEG, u'terriable': NEG, u'oppression': NEG, u'hatred': NEG, u'attack': NEG,u'fraud': NEG}
POS_Lexicon = {u'unity': POS, u'win': POS, u'winners': POS, u'pass': POS, u'freedom': POS, u'work': POS, u'peace': POS, u'good': POS, u'free': POS, u'liberty': POS, u'thank': POS, u'reform': POS, u'support': POS, u'better': POS, u'progress': POS, u'commitment': POS, u'dignity': POS, u'important': POS, u'courage': POS, u'well': POS,  u'strong': POS,  u'protect': POS,  u'honor': POS}
NET_Lexicon = {u'party': NET}

#Create an instance of SentimentIntesityAnalyzer called vader
vader = SentimentIntensityAnalyzer() #

#Adds terms and weights to the VADAR
vader.lexicon.update(NEG_Lexicon)
vader.lexicon.update(POS_Lexicon)
vader.lexicon.update(NET_Lexicon)

#Function that determines if the Category is either Republican, Democrat, Both or None
# Instead of comparing entire Title like in get_category, loop through each word and count the matches
# This version of get_category rewards an early mention and penalizes a later mention.  Increased Accuracy by 5-7%
def get_category(dataframe):
    results = []
    for Title in dataframe:  #Loop through all Titles in the dataframe
        R = 1  # Count Republican Category words
        D = 1 # Count Democrat Category words
        word_list = Title.split()  #Convert Title String into a list of words
        for i, word in enumerate (word_list):  #Loop through each word in the word_list, i is the index position of the word
            if any(ele in word for ele in R_Lexicon):
                R = R + (R / (i + len(word_list)))     # Divide by the index to reward mentions at the beginning of the headline and punish the end
            if any(ele in word for ele in D_Lexicon):
                D = D + (D / (i + len(word_list)))    # Divide by the index to reward mentions at the beginning of the headline and punish the end
            else:
                pass
        if R - D > -.001 and R - D < .001:  # Case when Both Republicans and Democrats are mentioned in the Tile.  Tune the value to determine how close they are.
            results.append("Both")
        elif R - D > 0:
            results.append("Republican")   #Case when Republicans are mentioned more often in the Title
        elif R - D < 0:
            results.append("Democrat")   # Case when Democrats are mentioned more often in the Title
        else: 
            results.append("None")      # Case where Neither party is mentioned in the Title
    return results

#Funtion that determines if the Title is Bias to left, right or netutral
def get_bias(df):
    results = []
    for index, row in df.iterrows():
        if (row["Sentiment"] == "negative" and row["Category"] == "Democrat"):
            row["Bias"]  = "Right"
        elif (row["Sentiment"] == "positive" and row["Category"] == "Democrat"):
            row["Bias"]  = "Left"
        elif row["Sentiment"] == "positive" and row["Category"] == "Republican":
            row["Bias"] = "Right"
        elif row["Sentiment"] == "negative" and row["Category"] == "Republican":
            row["Bias"]  = "Left"
        elif row["Sentiment"] == "neutral" and row["Category"] == "Republican":
            row["Bias"]  = "Right"
        elif row["Sentiment"] == "neutral" and row["Category"] == "Democrat":
            row["Bias"]  = "Left"
        elif row["Sentiment"] == "neutral" and row["Category"] == "Both":
            row["Bias"]  = "Neutral"
        elif row["Category"] == "Both":
            row["Bias"]  = "Neutral"
        else:
            row["Bias"]  = "Neutral"
        results.append(row["Bias"])
    return results

def measure_accuracy(Bias, PBias):
    results = []
    for B, P in zip(Bias, PBias):
        if B == P:
            results.append("TP")    # If the results match, it's a True Positive
        elif B == "Left" and P == "Right":
            results.append("FP")   #If it the Bias is Left but the predicaitoin is Right then it's a False Possative
        elif  B == "Right" and P == "Left":
            results.append("FN")   #If theBias is Right and the prediciton is Left then it's a False Negative
        elif B == "Neutral" and P == "Right":
            results.append("FP")   #If it the Bias is Neutral but the predicaitoin is Right then it's a False Possative
        elif  B == "Neutral" and P == "Left":
            results.append("FN")   #If the Bias is Neutral and the prediciton is Left then it's a False Negative
        else: 
            results.append("None")      # Case where Neither party is mentioned in the Title
    return results

#Calculate the Precision, Recall and F1 Scores
def get_accuracy(df):
    tp = (df.Results == "TP").sum()
    fp = (df.Results == "FP").sum()
    fn = (df.Results == "FN").sum()
    Precision = tp / (tp + fp)
    Recall = tp / (tp + fn)
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    Classification_Accuracy = tp / len(df)
    print ("Precision=",Precision)
    print ("Recall=",Recall)
    print ("F1=", F1)
    print ("Accuracy=",Classification_Accuracy)

if __name__ == "__main__":
    #Initialize Pandas Dataframe with test data
    df = pd.read_csv("CS410-BiasDetector/data_labeled/biasdetective2.csv")

    #Remove any extra quotes from the Titles for better matching
    df["Title"] = df["Title"].apply(lambda x: x.replace('"', ''))
    df["Title"] = df["Title"].apply(lambda x: x.replace("'", ''))
    df["Title"] = df["Title"].apply(lambda x: x.lower())

    #Add a new Sentiment score column to dataframe 
    df["Scores"] = df["Title"].apply(lambda Title: vader.polarity_scores(Title))

    #Add new compound value to the dataframe
    df["compound"] = df["Scores"].apply(lambda score_dict: score_dict["compound"])

    #Add a new Sentiment column for pos, neg or neutral
    df["Sentiment"] = df["compound"].apply(lambda c: "positive" if c > .02 else ("negative" if c < -.02 else "neutral"))

    #Add a new Category column to dataframe 
    df["Category"] = get_category(df["Title"])

    #Add a new Predicted Bias column to dataframe 
    df["Predicted_Bias"] = get_bias(df)

    #Get the accuracy
    df["Results"] = measure_accuracy(df["Bias"],df["Predicted_Bias"])

    #Calculate the overall accuracy of the model
    get_accuracy(df)

    #Export Results to CSV file
    df.to_csv("/Users/nbachman/Documents/HCP Anywhere/GradSchool/Text Mining and Analytics/CS410-BiasDetector/Rules_Model/results3.csv")