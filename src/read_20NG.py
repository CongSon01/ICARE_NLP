import pandas as pd
import re
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk

# Download necessary NLTK resources
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')

# Define stopwords
stop_words = set(stopwords.words('english'))

# Function to preprocess text (remove stopwords, punctuation)
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize text
    word_tokens = word_tokenize(text)
    # Remove stopwords and punctuation
    filtered_words = [
        word.lower() for word in word_tokens 
        if word.lower() not in stop_words and word not in string.punctuation
    ]
    return ' '.join(filtered_words)

# Function to load and preprocess dataset for a given subset
def load_and_preprocess_data(subset):
    categories = [
        'alt.atheism',
        'soc.religion.christian',
        'talk.religion.misc',
        'talk.politics.guns',
        'talk.politics.mideast',
        'talk.politics.misc'
    ]
    
    # Load the 20 Newsgroups dataset
    newsgroups_data = fetch_20newsgroups(
        subset=subset, 
        categories=categories, 
        remove=('headers', 'footers', 'quotes'), 
        shuffle=True, 
        random_state=42
    )
    
    # Create a DataFrame
    df = pd.DataFrame({'text': newsgroups_data.data, 'category': newsgroups_data.target})
    
    # Preprocess text and add to DataFrame
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Add label and question columns
    df['label'] = range(1, len(df) + 1)
    df['question'] = "Is this sentence Atheism, Religion, Guns, Mideast, or Politics?"
    
    # Select final columns
    df_final = df[['category', 'question', 'processed_text']]
    return df_final

# Main execution
if __name__ == '__main__':
    download_nltk_resources()  # Ensure required resources are downloaded
    
    # Load and preprocess training data
    df_train = load_and_preprocess_data('train')
    df_train.to_csv('./20NG/train.csv', index=False, header=False)  
    
    # Load and preprocess test data
    df_test = load_and_preprocess_data('test')
    df_test.to_csv('./20NG/test.csv', index=False, header=False) 

    print("Data saved to train.csv and test.csv successfully.")
