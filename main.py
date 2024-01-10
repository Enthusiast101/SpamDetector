import json
import customtkinter
import numpy as np
import tensorflow as tf
from text_processor import text_preprocess
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Loading model
model = tf.keras.models.load_model("spam_classifier_model.h5")

def processing(sentence):
    with open("variables.json", "r") as file:
        data = json.load(file)

    max_length, vocab_size = data["max_length"], data["vocab_size"]

    sentence = text_preprocess(sentence)
    encoded_words = [one_hot(sentence, vocab_size)]
    embedded_words = pad_sequences(encoded_words, padding="pre", maxlen=max_length)

    y_pred = model.predict(embedded_words)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    return y_pred[0][0]


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Spam Detector App")
        self.geometry(f"{500}x{360}")
        self.minsize(500, 360)

        self.grid_columnconfigure(0, weight=1)
        self._font = ("Montserrat", 15)

        # Getting message input
        self.textbox = customtkinter.CTkTextbox(master=self, width=250, border_width=2,
                                                border_color="#949A9F", font=self._font)
        self.textbox.grid(row=0, column=0, padx=(20, 20), pady=(20, 20), columnspan=2,sticky="nsew")

        # Printing Status
        self.message_box = customtkinter.CTkTextbox(master=self, width=250, height=35, activate_scrollbars=False,
                                                    border_width=2, border_color="#949a9f", font=self._font)
        self.message_box.grid(row=2, column=0, padx=(20, 20), pady=(0, 20), columnspan=2, sticky="nsew")
        self.message_box.insert("end", "Check Spam/ Ham")
        self.message_box.configure(state="disabled")

        # Check Button
        self.check = customtkinter.CTkButton(master=self, fg_color="transparent", height=40, text="CHECK",
                                                     border_width=2, font=("Righteous", 16), command=self.predict)
        self.check.grid(row=4, column=0, padx=(20, 20), pady=(0, 20), sticky="nsew")

        self.clear = customtkinter.CTkButton(master=self, fg_color="transparent", height=40, text="CLEAR",
                                             border_width=2, font=("Righteous", 16), command=self.clear)
        self.clear.grid(row=4, column=1, padx=(20, 20), pady=(0, 20), sticky="nsew")

    def predict(self):
        sentence = self.textbox.get("1.0", "end")

        if len(sentence.split()) != 0:
            self.message_box.configure(state="normal")
            self.message_box.delete("1.0", "end")
            self.message_box.insert("end", "Checking...")
            self.message_box.configure(fg_color="#1D1E1E")
            self.message_box.configure(state="disable")

            classify = processing(sentence)

            self.message_box.configure(state="normal")
            self.message_box.delete("1.0", "end")

            if classify == 0:
                self.message_box.insert("end", "Ham")
                self.message_box.configure(fg_color="#699561")
            else:
                self.message_box.insert("end", "Spam")
                self.message_box.configure(fg_color="#C53B33")

            self.message_box.configure(state="disabled")

        else:
            self.message_box.configure(state="normal")
            self.message_box.delete("1.0", "end")
            self.message_box.insert("end", "Message not found!")

    def clear(self):
        self.textbox.delete("1.0", "end")


if __name__ == "__main__":
    app = App()
    app.mainloop()
