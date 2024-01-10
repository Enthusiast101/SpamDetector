import json
import numpy as np
import pandas as pd
from text_processor import text_preprocess
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def main():
    global max_length
    df = pd.read_csv("emails.csv")

    df = df.sample(frac=1)

    X, y = df["text"], df["spam"]
    X, y = X.to_frame(), y.to_frame()

    X["new_text"] = X["text"].apply(text_preprocess)

    vocab_size = 10000
    encoded_words = [one_hot(sentences, vocab_size) for sentences in X["new_text"]]

    max_length = len(max(X["new_text"]))
    embedded_words = pad_sequences(encoded_words, padding="pre", maxlen=max_length)

    with open("variables.json", "w") as file:
        variables = {
            "max_length": 581,
            "vocab_size": vocab_size
        }

        json.dump(variables, file)

    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=max_length, trainable=False))
    model.add(LSTM(units = 128, return_sequences = True, recurrent_dropout = 0.3, dropout = 0.5))
    model.add(LSTM(units = 64, recurrent_dropout = 0.3, dropout = 0.5))
    model.add(Dense(units = 32, activation = "relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    X_final, y_final = np.array(embedded_words), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_state=42)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

    model.save("spam_classifier_model.h5")


if __name__ == "__main__":
    if tf.test.is_gpu_available(cuda_only=True):
        with tf.device("/GPU:0"):
            main()
    else:
        main()
