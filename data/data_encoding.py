# This file shows how data encoding was done.
import pandas as pd

df = pd.read_csv("../data/HAM10000_metadata.csv")
classes = sorted(df["dx"].unique())
label2id = {label: idx for idx, label in enumerate(classes)}
# id2label = {k: v for v, k in label2id.items()} # Reverts the encoding
df['label'] = df['dx'].map(label2id)
df.to_csv("enc_HAM10000_metadata.csv", index=False)