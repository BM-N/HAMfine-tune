import os
from sklearn.model_selection import train_test_split
import pandas as pd

# Full data split
df = pd.read_csv(os.path.abspath("enc_HAM10000_metadata.csv"))

# train and test split
x_tmp_train, x_test, y_tmp_train, y_test = train_test_split(df['image_id'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)

test_set = pd.DataFrame(data={'image_id': x_test, 'label': y_test})
test_set.to_csv("test_set.csv")

# temporary df
temp_df = pd.DataFrame(data={'image_id': x_tmp_train, 'label': y_tmp_train})

# train and val split
x_train, x_val, y_train, y_val = train_test_split(temp_df['image_id'], temp_df['label'], test_size=0.25, stratify=temp_df['label'], random_state=42)

train_set = pd.DataFrame(data={'image_id': x_train, 'label': y_train})
train_set.to_csv("train_set.csv")

val_set = pd.DataFrame(data={'image_id': x_val, 'label': y_val})
val_set.to_csv("val_set.csv")