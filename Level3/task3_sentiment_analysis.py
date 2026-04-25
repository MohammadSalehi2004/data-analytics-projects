# Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Its my first time using NLTK so i will explain this part with more comments for my own future understanding
import nltk

# imports stopwords, which is common words like “the”, “is”, “and”
from nltk.corpus import stopwords
# imports tokenize which which splits a sentence into seperate words
from nltk.tokenize import word_tokenize
# imports lemmatization, which turns words to their base form so for example working "would" be "work"
from nltk.stem import WordNetLemmatizer

# textBlob is used to calculate sentiment polarity and imports wordcloud to create a word frequency image
from textblob import TextBlob
from wordcloud import WordCloud


# downloading the required nltk tools
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt_tab")

# using this for cleaner syling of plot
sns.set_theme()

# The path for the sentiment dataset
input_file = "Data/Sentiment dataset.csv"

# Loading dataset
df = pd.read_csv(input_file)

# showing some information about the dataset
print("First 5 rows of the dataset:")
print(df.head(), "\n")

print("Dataset information:")
df.info()
print()

print("Missing values in each column:")
print(df.isnull().sum(), "\n")


# cleaning the data
df = df.dropna(subset=["Text"])

# setting up stopwords and lemmatizer

# this creates a set of of english stopwords to remove from text
stop_words = set(stopwords.words("english"))
# creats the lemmatizer object
lemmatizer = WordNetLemmatizer()


# preprocessing text by tokenizing, removing stopwords, and lemmatizing
def preprocess_text(text):
    text = str(text).lower()
    tokens = word_tokenize(text)

    cleaned_words = []

    for word in tokens:
        if word.isalpha() and word not in stop_words:
            lemmatized_word = lemmatizer.lemmatize(word)
            cleaned_words.append(lemmatized_word)

    return " ".join(cleaned_words)


# applying preprocessing to the text column
df["Cleaned_Text"] = df["Text"].apply(preprocess_text)


# creating a function to classify sentiment using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


# applying sentiment analysis
df["Predicted_Sentiment"] = df["Cleaned_Text"].apply(get_sentiment)


# counting each sentiment
sentiment_counts = df["Predicted_Sentiment"].value_counts()

print("Sentiment Distribution:")
print(sentiment_counts, "\n")


# creating a bar chart for sentiment distribution
plt.figure()

plt.bar(sentiment_counts.index, sentiment_counts.values)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("Level3/Plots/sentiment_distribution.png")
plt.close()

print("Saved: sentiment_distribution.png")


# combining all cleaned text for word frequency visualization
all_words = " ".join(df["Cleaned_Text"])

# creating a word cloud for word frequencies
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_words)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Frequencies")

plt.tight_layout()

plt.savefig("Level3/Plots/word_frequencies.png")
plt.close()

print("Saved: word_frequencies.png")