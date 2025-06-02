import nltk  # module: natural language
import re   # regex: re.search
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt')   # 
nltk.download('stopwords')   #  The product is not good
nltk.download('wordnet')   # lemmatization: 

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))   # {the, is, a}\
print(stop_words)
lemmatizer = WordNetLemmatizer()   #

def preprocess_text(text):  # The shops product is really good
    # Lowercase the text
    text = text.lower()  # 
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)  # 

    # Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text)
    
    # theshopsproductisreallygood
    # Tokenize
    words_to_keep = {
    'not', 'no', 'nor', "don't", "didn't", "won't", "can't", "isn't",
    "wasn't", "weren't", "couldn't", "shouldn't", "wouldn't", "haven't",
    "hasn't", "hadn't", "aren't", "ain", "mustn't", "needn't"
    }
    custom_stopwords = stop_words - words_to_keep
    tokens = nltk.word_tokenize(text)  # ['the','shops','product','is','really','good']
    
    # Remove stopwords and short tokens, and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords and len(word) > 2]
    
    # ['shops','product','really','good']   str=' '  str.join(tokens)


    # Return cleaned sentence
    return ' '.join(tokens)  # shops product really good
