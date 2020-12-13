import nltk
import ssl
import numpy as np
import pandas as pd
import difflib as dfl
import spacy
nlp = spacy.load("en_core_web_sm")

#Only need to run first time code the first time
try:
    _create_unverified_https_category = ssl._create_unverified_category
except AttributeError:
    pass
else:
    ssl._create_default_https_category = _create_unverified_https_category

nltk.download("vader_lexicon") # downloads vader_lexicon to /nltk_data
print("import nltk,vader and SentimentIntesity complete")

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

def measure_accuracy(Bias, PBias, Category):
    right_results = []
    left_results = []
    neutral_results = []

    for B, P in zip(Bias, PBias):
        if B == "Left" and P == "Left":  # Case when you correctly predicted the Left Category
            left_results.append("TP") 
            right_results.append("TN")
            neutral_results.append("TN") 
        elif  B == "Left" and P == "Right":  # Case when you falsely predicted the Right Category when it was Left
            left_results.append("FN") 
            right_results.append("FP")
            neutral_results.append("TN")  
        elif  B == "Left" and P == "Netural":  # Case when you falsely predicted the Neutral Category when it was Left
            left_results.append("FN") 
            right_results.append("TN")
            neutral_results.append("FP")
        elif  B == "Right" and P == "Right":  # Case when you correctly predicted it was Right
            left_results.append("TN") 
            right_results.append("TP")
            neutral_results.append("TN")
        elif  B == "Right" and P == "Left":  # Case when you falsely predicted it was Left what it was Right
            left_results.append("FP") 
            right_results.append("FN")
            neutral_results.append("TN")
        elif  B == "Right" and P == "Neutral":  # Case when you falsely predicted the Neutral Category when it was Right
            left_results.append("TN") 
            right_results.append("FN")
            neutral_results.append("FP")  
        elif  B == "Neutral" and P == "Neutral":  # Case when you correctly predicted it was Neutral
            left_results.append("TN") 
            right_results.append("TP")
            neutral_results.append("TN")
        elif  B == "Neutral" and P == "Left":  # Case when you falsely predicted it was Left what it was Neutral
            left_results.append("FP") 
            right_results.append("TN")
            neutral_results.append("FN")
        elif  B == "Neutral" and P == "Right":  # Case when you falsely predicted the Right Category when it was Neutral
            left_results.append("TN") 
            right_results.append("FP")
            neutral_results.append("FN") 
        else: 
            left_results.append("None") 
            right_results.append("None")
            neutral_results = ("None") 
    
    if Category == "R":
        return right_results
    if Category == "L":
        return left_results
    if Category == "N":
        return neutral_results
    else:
        pass

#Calculate the Precision, Recall and F1 Scores
def get_accuracy(df):
    r_tp = (df.Right_Results == "TP").sum()
    r_fp = (df.Right_Results == "FP").sum()
    r_fn = (df.Right_Results == "FN").sum()
    R_Precision = r_tp / (r_tp + r_fp)
    R_Recall = r_tp / (r_tp + r_fn)
    R_f1 = (2 * R_Precision * R_Recall) / (R_Precision + R_Recall)
    print ("R_Precision=",R_Precision)
    print ("R_Recall=",R_Recall)
    print ("R_F1=", R_f1)

    l_tp = (df.Left_Results == "TP").sum()
    l_fp = (df.Left_Results == "FP").sum()
    l_fn = (df.Left_Results == "FN").sum()
    L_Precision = l_tp / (l_tp + l_fp)
    L_Recall = l_tp / (l_tp + l_fn)
    L_f1 = (2 * L_Precision * L_Recall) / (L_Precision + L_Recall)

    print ("L_Precision=",L_Precision)
    print ("L_Recall=",L_Recall)
    print ("L_F1=", L_f1)

if __name__ == "__main__":
    #Initialize Pandas Dataframe with test data
    df = pd.read_csv("CS410-BiasDetector/data_labeled/biasdetective1.csv")

    #Remove any extra quotes from the Titles for better matching
    df["Title"] = df["Title"].apply(lambda x: x.replace('"', ''))
    df["Title"] = df["Title"].apply(lambda x: x.replace("'", ''))
    df["Title"] = df["Title"].apply(lambda x: x.lower())

    #Add a new Sentiment score column to dataframe 
    df["Scores"] = df["Title"].apply(lambda Title: vader.polarity_scores(Title))

    #Add new Compound value to the dataframe
    df["compound"] = df["Scores"].apply(lambda score_dict: score_dict["compound"])

    #Add a new Sentiment column for pos, neg or neutral
    df["Sentiment"] = df["compound"].apply(lambda c: "positive" if c > .02 else ("negative" if c < -.02 else "neutral"))

    #Add a new Category column to dataframe 
    df["Category"] = get_category(df["Title"])

    #Add a new Predicted Bias column to dataframe 
    df["Predicted_Bias"] = get_bias(df)

    #Get the accuracy of the Right Category
    df["Right_Results"] = measure_accuracy(df["Bias"],df["Predicted_Bias"], "R")

    #Get the accuracy of the Left Category
    df["Left_Results"] = measure_accuracy(df["Bias"],df["Predicted_Bias"], "L")

    #Get the accuracy of the Neutral Category
    df["Neutral_Results"] = measure_accuracy(df["Bias"],df["Predicted_Bias"], "N")

    #Calculate the accuracy of the model using Precision, Recall and F1 per category
    get_accuracy(df)

    #Export Results to CSV file
    df.to_csv("results1.csv")