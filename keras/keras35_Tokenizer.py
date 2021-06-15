from tensorflow.keras.preprocessing.text import Tokenizer

text = "I ate really really delcious food"

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)

x = token.texts_to_sequences([text])
print(x)
# Changing Data to Dictionanry and make words to numeric data.

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size)
x = to_categorical(x)
print(x)
print(x.shape) # to_categorical starts with 0 but this token statrts with 1
               # so shape is not (6,5) but (6,6)