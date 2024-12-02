import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# ãƒˆãƒ¼ã‚¯ãƒŠã‚¤ã‚¶ãƒ¼ã¨ãƒ¢ãƒ‡ãƒ«ã®èª­ã¿è¾¼ã¿
tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v2")
model = AutoModel.from_pretrained("tohoku-nlp/bert-base-japanese-v2")

# ãƒ‡ãƒ¼ã‚¿ã®èª­ã¿è¾¼ã¿
df = pd.read_csv("data/日経新聞/keitaiso_nikkei_1212-2212.csv")

# 2016å¹´ä»¥é™ã®ãƒ‡ãƒ¼ã‚¿ã«ãƒ•ã‚£ãƒ«ã‚¿ãƒªãƒ³ã‚°
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df[df['date'].dt.year >= 2016]

# å¿…è¦ãªã‚«ãƒ©ãƒ ã®ã¿ã‚’ä¿æŒ
df = df[['date', 'wakachi', 'tokenizer']]

def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    return torch.mean(last_hidden_state, dim=1).squeeze().numpy()

# å„æ–‡ã‚’768æ¬¡å…ƒã®ãƒ™ã‚¯ãƒˆãƒ«ã«å¤‰æ›
df['bert_vector'] = df['tokenizer'].apply(lambda x: encode_text(x, tokenizer, model))

# æ—¥åˆ¥ã«ãƒ™ã‚¯ãƒˆãƒ«ã‚’å¹³å‡
daily_vectors = df.groupby(df['date'].dt.date)['bert_vector'].apply(lambda x: np.mean(np.stack(x), axis=0))

# ãƒ‡ãƒ¼ã‚¿ãƒ•ãƒ¬ãƒ¼ãƒ ã¨ã—ã¦ä¿å­˜
daily_vectors_df = pd.DataFrame(daily_vectors.tolist(), index=daily_vectors.index, columns=[f'feature_{i}' for i in range(768)])
daily_vectors_df.reset_index(inplace=True)
daily_vectors_df.rename(columns={'index': 'date'}, inplace=True)

# ä¿å­˜
daily_vectors_df.to_csv("daily_vectors.csv", index=False)