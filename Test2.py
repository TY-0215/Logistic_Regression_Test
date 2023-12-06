import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent(X, y, learning_rate=0.05, epochs=1000):
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]  # 편향(bias)을 위해 X에 1 추가
    W = np.random.rand(n + 1)
    b = np.random.rand()

    for i in range(epochs):
        z = np.dot(X, W) + b
        h = sigmoid(z)
        error = np.abs(h - y).mean()

        if error < 0.001:
            break

        W_grad = learning_rate * np.dot((h - y), X) / m
        b_grad = learning_rate * (h - y).mean()

        W = W - W_grad
        b = b - b_grad

    return W, b

def predict(X, W, b):
    X = np.c_[np.ones((X.shape[0], 1)), X]
    predictions = sigmoid(np.dot(X, W) + b)
    return predictions

# 데이터 로드
data = pd.read_csv("c:/Users/dsem/Desktop/Py_Test/weatherReports.csv")

# 필요한 이벤트 유형 선택 (Thunderstorm Wind 및 Hail)
selected_events = ['Thunderstorm Wind', 'Hail']
selected_data = data.loc[data['event_type'].isin(selected_events), ['event_narrative', 'event_type']].copy()

# 누락된 값을 빈 문자열로 대체
selected_data['event_narrative'].fillna("", inplace=True)

# 입력 데이터와 라벨 데이터 선택
X = selected_data['event_narrative']
y = (selected_data['event_type'] == 'Thunderstorm Wind').astype(int)  # 이진 분류

# Bag-of-Words로 텍스트 특징 추출
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# 훈련 데이터와 테스트 데이터 분할 (랜덤하게 7:3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 내리막 경사법으로 로지스틱 회귀 모델 학습
W, b = gradient_descent(X_train.toarray(), y_train)

# 테스트 데이터로 예측
predictions = predict(X_test.toarray(), W, b)
y_pred = (predictions >= 0.5).astype(int)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")