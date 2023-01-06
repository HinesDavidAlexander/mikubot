import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        self.categorization_values = {
            '_positive': [0.7, 1.0], 
            '_slightly_positive': [0.4, 0.7], 
            '_neutral': [-0.35, 0.4], 
            '_slightly_negative': [-0.7, -0.35], 
            '_negative': [-1.0, -0.7]
        }

    def categorize(self, scores):
        for folder, range_values in self.categorization_values.items():
            if (scores < range_values[1]) and (scores > range_values[0]):
                return folder
        # if scores['compound'] > 0.5:
        #     return "_positive", scores
        # elif scores['compound'] < -0.5:
        #     return "_negative", scores
        # else:
        #     return "_neutral", scores

    def analyze(self, text: str, mode: int = 0):
        if mode == 0:
            return self.sent_nltk(text)
        elif mode == 1:
            return self.sent_blob(text)
        else:
            return "please enter a valid mode. 0,1", 0.0

    def sent_nltk(self, text: str):
        scores = self.sid.polarity_scores(text)
        return self.categorize(scores['compound'])
    
    def sent_blob(self, text: str):
        text_blob = TextBlob(text)
        return self.categorize(text_blob.sentiment.polarity)

if __name__ == '__main__':
    #nltk.download('vader_lexicon')
    text = "My house is nice"
    analyzer = SentimentAnalyzer()
    val = analyzer.analyze(text, 1)
    print(val)
    print('done')