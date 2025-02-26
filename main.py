import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os
from dotenv import load_dotenv
import kagglehub
from tqdm import tqdm
from joblib import dump, load
from imblearn.over_sampling import SMOTE  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import gensim
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Sử dụng Keras chỉ cho việc tokenize và padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
load_dotenv()


def check_gpu():
    """Kiểm tra và in thông tin các GPU khả dụng."""
    if torch.cuda.is_available():
        print("GPU is available:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("GPU is not available. Running on CPU.")


class TwitterClient(object):
    def __init__(self):
        pass

    def clean_tweet(self, tweet):
        return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        # Đoạn code sử dụng TextBlob ban đầu được thay thế bởi mô hình học sâu.
        pass


def load_tweets_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin1', header=None)
        df.columns = ["sentiment", "id", "date", "query", "user", "text"]
        df["sentiment"] = df["sentiment"].replace({0: -1, 4: 1})

        tweets = []
        for index, row in df.iterrows():
            parsed_tweet = {"text": row["text"], "sentiment": row["sentiment"]}
            tweets.append(parsed_tweet)
        return tweets
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return None


def preprocess_text(text, use_lemmatization=True):
    text = text.lower()  # Chuyển về chữ thường
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def train_word2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=4):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, num_classes, embedding_matrix, max_len):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Không cập nhật trọng số embedding
        self.lstm = nn.LSTM(embedding_dim, lstm_units, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(lstm_units * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, (hn, cn) = self.lstm(x)
        forward_hidden = hn[-2, :, :]
        backward_hidden = hn[-1, :, :]
        hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out


def train_and_evaluate_bilstm(tweets, embedding_dim=100, lstm_units=128, dropout_rate=0.2, max_len=50,
                              model_path="models/BiLSTM_Word2Vec_model.pt", tokenizer_path="models/tokenizer_and_encoder.joblib",
                              batch_size=16, num_epochs=10):
    texts = [tweet['text'] for tweet in tweets]
    sentiments = [tweet['sentiment'] for tweet in tweets]

    processed_texts = []
    for text in tqdm(texts, desc="Preprocessing text"):
        processed_texts.append(preprocess_text(text))

    print("Training Word2Vec model...")
    tokenized_sentences = [text.split() for text in processed_texts]
    word2vec_model = train_word2vec_model(tokenized_sentences, vector_size=embedding_dim)
    vocab_size = len(word2vec_model.wv.key_to_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i, word in enumerate(word2vec_model.wv.key_to_index):
        embedding_matrix[i + 1] = word2vec_model.wv[word]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts(processed_texts)
    sequences = tokenizer.texts_to_sequences(processed_texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    label_encoder = LabelEncoder()
    y = np.array(sentiments)
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y_encoded, test_size=0.2, random_state=2, stratify=y_encoded)

    smote = SMOTE(random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = len(label_encoder.classes_)
    model = BiLSTMModel(vocab_size, embedding_dim, lstm_units, num_classes, embedding_matrix, max_len)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Huấn luyện mô hình với tqdm hiển thị tiến trình của từng batch
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
            for batch_X, batch_y in tepoch:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
                tepoch.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Đánh giá mô hình: sử dụng DataLoader cho tập test để tránh lỗi bộ nhớ GPU
    model.eval()
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predictions = []

    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    y_test_decoded = label_encoder.inverse_transform(y_test_tensor.cpu().numpy())
    y_pred_decoded = label_encoder.inverse_transform(np.array(predictions))

    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    report = classification_report(y_test_decoded, y_pred_decoded, labels=[-1, 0, 1], zero_division=0)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    dump((tokenizer, label_encoder, word2vec_model), tokenizer_path)
    print(f"Tokenizer and Encoder saved to {tokenizer_path}")

    return accuracy, report


def load_model(model_path="models/BiLSTM_Word2Vec_model.pt", tokenizer_path="models/tokenizer_and_encoder.joblib",
               embedding_dim=100, lstm_units=128, max_len=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer, label_encoder, word2vec_model = load(tokenizer_path)
        print(f"Tokenizer and encoder loaded from {tokenizer_path}")
    except Exception as e:
        print(f"Error loading tokenizer and encoder: {e}")
        return None, None, None, None

    vocab_size = len(word2vec_model.wv.key_to_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i, word in enumerate(word2vec_model.wv.key_to_index):
        embedding_matrix[i + 1] = word2vec_model.wv[word]

    num_classes = len(label_encoder.classes_)
    model = BiLSTMModel(vocab_size, embedding_dim, lstm_units, num_classes, embedding_matrix, max_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"Model loaded from {model_path}")
    return model, tokenizer, label_encoder, word2vec_model


def main():
    check_gpu()  # Kiểm tra GPU ngay từ đầu

    api = TwitterClient()

    dataset_path = kagglehub.dataset_download("kazanova/sentiment140")
    csv_file = os.path.join(dataset_path, "training.1600000.processed.noemoticon.csv")
    print("Data path:" + str(csv_file))
    print("Chạy thử nghiệm với dữ liệu từ file csv")

    tweets = load_tweets_from_csv(csv_file)
    if not tweets:
        print("No tweets found.")
        return

    print(f"Số lượng tweet đã lấy: {len(tweets)}")
    print("\nRunning experiment with BiLSTM and Word2Vec using PyTorch")

    accuracy, report = train_and_evaluate_bilstm(tweets)
    if accuracy is not None:
        print(f"  Model: BiLSTM with Word2Vec (PyTorch)")
        print(f"  Accuracy: {accuracy:.2f}")
        print(f"  Classification Report:\n{report}")

    loaded_model, tokenizer, label_encoder, word2vec_model = load_model()
    if loaded_model and tokenizer and label_encoder and word2vec_model:
        print("\nModel and Tokenizer loaded successfully!")
        sample_text = "This is an amazing movie! I really enjoyed it."
        processed_text = preprocess_text(sample_text)
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=50, padding='post', truncating='post')

        input_tensor = torch.tensor(padded_sequence, dtype=torch.long).to("cuda" if torch.cuda.is_available() else "cpu")

        loaded_model.eval()
        with torch.no_grad():
            outputs = loaded_model(input_tensor)
            _, prediction = torch.max(outputs, 1)
            prediction = prediction.cpu().numpy()
        predicted_sentiment = label_encoder.inverse_transform(prediction)[0]
        print(f"Sample text: {sample_text}")
        print(f"Predicted sentiment: {predicted_sentiment}")
    else:
        print("Failed to load model or tokenizer.")


if __name__ == "__main__":
    main()
