
import os
import torch
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pandas as pd


MAX_LEN = 300
data_file_address = "data.json"
models_paths = [path for path in os.listdir('models') if path.endswith('.torch')]

# IZBERI MODEL
print("Izberi model za evaluacijo: ")
for i, file in enumerate(models_paths):
    print(f'{i}. {file}')
i = int(input("Enter models index to evaluate: "))
model_path = models_paths[i]

# LOAD MODEL
model = torch.load(f'models/{model_path}')
model.eval()

# LOAD TEXT
df_data = pd.read_json(data_file_address, lines=True)
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(df_data["content"])
tokenized_texts = tokenizer.texts_to_sequences("Hello World")
input_ids = pad_sequences(tokenized_texts,
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
input_tensor = torch.tensor(input_ids)
with torch.no_grad():
    output = model(input_tensor)
    print(output)