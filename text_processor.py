import re
import nltk

nltk.download('punkt')
nltk.download("stopwords")


def text_preprocess(sentence):
    # Tokenization
    tokens = nltk.word_tokenize(sentence)

    # Symbol Removal
    n_corpus = []
    for i in range(len(tokens)):
        review = re.sub(r"[^a-zA-Z]", " ", tokens[i])
        review = review.lower()
        n_corpus.append(review)

    # Stopword Removal
    stopwords = nltk.corpus.stopwords.words("english")
    rem_stopwords = [words for words in n_corpus if words not in stopwords and
                     len(words.split()) != 0 and
                     len(words) > 2]

    # Lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    transformed_words = [lemmatizer.lemmatize(words) for words in rem_stopwords]

    return " ".join(transformed_words)
