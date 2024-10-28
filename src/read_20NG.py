import pandas as pd
import re
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk

def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\d+', '', text)
    word_tokens = word_tokenize(text)
    filtered_words = [
        word.lower() for word in word_tokens 
        if word.lower() not in stop_words and word not in string.punctuation
    ]
    return ' '.join(filtered_words)

def load_and_preprocess_data(subset):
    categories = [
        'alt.atheism',
        'soc.religion.christian',
        'talk.religion.misc',
        'talk.politics.guns',
        'talk.politics.mideast',
        'talk.politics.misc'
    ]
    
    newsgroups_data = fetch_20newsgroups(
        subset=subset, 
        categories=categories, 
        remove=('headers', 'footers', 'quotes'), 
        shuffle=True, 
        random_state=42
    )
    df = pd.DataFrame({'text': newsgroups_data.data, 'category': newsgroups_data.target})
    
    df['processed_text'] = df['text'].apply(preprocess_text)
    df['label'] = range(1, len(df) + 1)
    df['question'] = "Is this sentence Atheism, Religion, Guns, Mideast, or Politics?"
    
    df_final = df[['category', 'question', 'processed_text']]
    return df_final

# Main execution
if __name__ == '__main__':
    download_nltk_resources()  
    
    df_train = load_and_preprocess_data('train')
    df_train.to_csv('./20NG/train.csv', index=False, header=False)  
    
    df_test = load_and_preprocess_data('test')
    df_test.to_csv('./20NG/test.csv', index=False, header=False) 

    print("Data saved to train.csv and test.csv successfully.")
