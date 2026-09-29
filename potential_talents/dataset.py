from pathlib import Path
import pandas as pd
#data preprocessing tools
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def load_dataset(path) -> pd.DataFrame:
    """Load and validate the processed candidate dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)

    return data

def clean_text(data) -> pd.DataFrame:
    """Clean the job title & location columns."""
    #convert column to lowercase
    data['job_title_clean'] = data['job_title'].str.lower()
    data['location_clean'] = data['location'].str.lower()

    #remove puncuation and whitespace
    data['job_title_clean'] = data['job_title_clean'].str.replace(r'[^\w\s]', '', regex=True) 
    data['location_clean'] = data['location_clean'].str.replace(r'[^\w\s]', '', regex=True) 

    #remove numerics
    data['job_title_clean'] = data['job_title_clean'].str.replace(r'\d+', '', regex=True) 

    #remove accents from characters
    data['location_clean'] = data['location_clean'].str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    return data

def create_tokens(data) -> pd.DataFrame:
    """Clean the candidate dataset."""

    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return tokens

    data['tokens'] = data['job_title_clean'].apply(preprocess)
    return data

def save_dataset(data, name):
    """Save the processed candidate dataset."""
    data.to_csv(f"../data/processed/{name}.csv", index=False) 