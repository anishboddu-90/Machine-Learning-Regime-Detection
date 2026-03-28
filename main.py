import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

def generate_dataset(steps):
    dataset = pd.DataFrame(columns=['steps','cumsum5','cumsum10','cumsum20','volatility','streak'])
    steps_list = []
    bias_active = False
    bias_states_list = []
    bias_states = pd.DataFrame(columns=['bias_active'])
    bias_duration_remaining = 0
    for _ in range(steps):
        if random.random() < 0.02 and bias_active == False:
                bias_active = True
                bias_duration_remaining = 16
        if bias_duration_remaining > 0:
            bias_active = True
            bias_duration_remaining -= 1
        else:
            bias_active = False
        if bias_active == True:
            step = random.choices((1,-1), weights = [0.48,0.52])[0]
        else:
            step = random.choice([1,-1])
        steps_list.append(step)
        bias_states_list.append(bias_active)
    dataset['steps'] = steps_list
    bias_states['bias_active'] = bias_states_list
    return dataset, bias_states

steps, labels = generate_dataset(10000)

def feature_generation(df):
    df['cumsum5']=df['steps'].rolling(window=5, min_periods=1).sum()
    df['cumsum10']=df['steps'].rolling(window=10, min_periods=1).sum()
    df['cumsum20']=df['steps'].rolling(window=20, min_periods=1).sum()
    df['volatility']=df['steps'].rolling(window=5, min_periods=1).std().fillna(0)
    df['streak1']=df['steps'].groupby((df['steps'] !=1).cumsum()).cumsum().where(df['steps'] == 1, 0)
    df['streak-1']=df['steps'].groupby((df['steps'] !=-1).cumsum()).cumsum().where(df['steps'] == -1, 0)
    df['autocorr20']=df['steps'].rolling(window=20, min_periods=1).apply(lambda x: x.autocorr(lag=1), raw=False).fillna(0)
    df['autocorr5']=df['steps'].iloc[:].rolling(window=5, min_periods=1).apply(lambda x: x.autocorr(lag=1), raw=False).fillna(0)
    return df

features = feature_generation(steps)
features['label'] = labels['bias_active']
features['label'] = features['label'].shift(-1)
features = features.dropna(subset=['label'])
X = features.drop(columns=['label', 'steps'])
y = features['label'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5, class_weight='balanced')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(np.mean(y))