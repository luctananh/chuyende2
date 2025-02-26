import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump, load
from dotenv import load_dotenv

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import gensim
from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Dùng Keras cho tokenize và padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import kagglehub

# Tải các tài nguyên cần thiết từ NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

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
        # Loại bỏ mentions, URL và ký tự đặc biệt bằng regex
        return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|(\w+:\/\/\S+)|([^0-9A-Za-z \t])", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        # Đoạn code ban đầu dùng TextBlob đã được thay bằng mô hình học sâu.
        pass


def load_tweets_from_csv(file_path):
    try:
        df = pd.read_csv(file_path, encoding='latin1', header=None)
        df.columns = ["sentiment", "id", "date", "query", "user", "text"]
        # Chỉ có 2 lớp: 0 (âm) và 4 (dương); đổi 0 -> -1, 4 -> 1
        df["sentiment"] = df["sentiment"].replace({0: -1, 4: 1})
        
        tweets = []
        for _, row in df.iterrows():
            parsed_tweet = {"text": row["text"], "sentiment": row["sentiment"]}
            tweets.append(parsed_tweet)
        return tweets
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
        return None


def preprocess_text(text, use_lemmatization=True, remove_stopwords=True):
    """
    Tiền xử lý text:
      - Chuyển về chữ thường
      - Loại bỏ URL, mentions, ký tự đặc biệt
      - Giữ lại từ sau dấu #
      - (Tùy chọn) Loại bỏ stopwords và thực hiện lemmatization
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokenizer = TweetTokenizer()
    words = tokenizer.tokenize(text)
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        
    return ' '.join(words)


def train_word2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=30):
    """
    Huấn luyện Word2Vec với skip-gram (sg=1) và số epoch cao hơn.
    """
    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=1)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    return model


# Lớp attention đơn giản
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, lstm_outputs):
        # lstm_outputs: [batch_size, seq_len, hidden_size*2]
        weights = torch.tanh(self.attention(lstm_outputs))  # [batch_size, seq_len, 1]
        weights = weights.squeeze(-1)  # [batch_size, seq_len]
        weights = torch.softmax(weights, dim=1)  # [batch_size, seq_len]
        # Tính weighted sum
        weighted_output = torch.bmm(weights.unsqueeze(1), lstm_outputs).squeeze(1)  # [batch_size, hidden_size*2]
        return weighted_output, weights


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, num_classes, embedding_matrix, max_len, freeze_embeddings=False):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Khởi tạo embedding từ embedding_matrix
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        # Cho phép cập nhật trọng số embedding nếu cần (fine-tuning)
        self.embedding.weight.requires_grad = not freeze_embeddings
        self.lstm = nn.LSTM(embedding_dim, lstm_units, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.5)  # tăng dropout
        self.attention = Attention(lstm_units)
        self.dropout = nn.Dropout(0.5)  # tăng dropout
        self.fc = nn.Linear(lstm_units * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, lstm_units*2]
        attn_output, attn_weights = self.attention(lstm_out)
        attn_output = self.dropout(attn_output)
        out = self.fc(attn_output)
        return out


def train_and_evaluate_bilstm(tweets, embedding_dim=100, lstm_units=128, max_len=50,
                              model_path="models/BiLSTM_Word2Vec_model.pt", 
                              tokenizer_path="models/tokenizer_and_encoder.joblib",
                              batch_size=32, num_epochs=20, learning_rate=0.001):
    texts = [tweet['text'] for tweet in tweets]
    sentiments = [tweet['sentiment'] for tweet in tweets]

    # Tiền xử lý từng tweet
    processed_texts = []
    for text in tqdm(texts, desc="Preprocessing text"):
        processed_texts.append(preprocess_text(text))

    print("Training Word2Vec model...")
    tokenized_sentences = [text.split() for text in processed_texts]
    word2vec_model = train_word2vec_model(tokenized_sentences, vector_size=embedding_dim, epochs=30)
    
    # Xây dựng từ điển với Tokenizer của Keras
    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(processed_texts)
    sequences = tokenizer.texts_to_sequences(processed_texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    # Xây dựng embedding matrix dựa trên từ vựng của Tokenizer và embedding của Word2Vec
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]
    
    label_encoder = LabelEncoder()
    y = np.array(sentiments)
    y_encoded = label_encoder.fit_transform(y)

    # Chia dữ liệu thành tập train và test theo tỷ lệ 80/20, stratify theo nhãn
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y_encoded, test_size=0.2,
                                                        random_state=2, stratify=y_encoded)
    
    # Tính trọng số lớp (class weights) cho dữ liệu không cân bằng
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Chuyển đổi dữ liệu sang tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print(f"Using device: {device}")

    num_classes = len(label_encoder.classes_)
    model = BiLSTMModel(vocab_size, embedding_dim, lstm_units, num_classes, embedding_matrix, max_len,
                        freeze_embeddings=False)  # Cho phép fine-tuning embedding
    model = model.to(device)

    # Định nghĩa loss function với trọng số lớp và optimizer
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Huấn luyện mô hình
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
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    # Đánh giá mô hình trên tập test
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
    report = classification_report(y_test_decoded, y_pred_decoded, labels=[-1, 1], zero_division=0)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")

    # Lưu mô hình và đối tượng tokenizer/encoder
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

    # Xây dựng embedding matrix từ tokenizer.word_index và word2vec model
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    num_classes = len(label_encoder.classes_)
    model = BiLSTMModel(vocab_size, embedding_dim, lstm_units, num_classes, embedding_matrix, max_len,
                        freeze_embeddings=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"Model loaded from {model_path}")
    return model, tokenizer, label_encoder, word2vec_model


def main():
    check_gpu()  # Kiểm tra GPU ngay từ đầu

    api = TwitterClient()

    dataset_path = kagglehub.dataset_download("kazanova/sentiment140")
    csv_file = os.path.join(dataset_path, "training.1600000.processed.noemoticon.csv")
    print("Data path: " + str(csv_file))
    print("Chạy thử nghiệm với dữ liệu từ file csv")

    tweets = load_tweets_from_csv(csv_file)
    if not tweets:
        print("No tweets found.")
        return

    print(f"Số lượng tweet đã lấy: {len(tweets)}")
    print("\nRunning experiment with BiLSTM and Word2Vec using PyTorch with Attention")

    accuracy, report = train_and_evaluate_bilstm(tweets)
    if accuracy is not None:
        print(f"  Model: BiLSTM with Word2Vec and Attention (PyTorch)")
        print(f"  Accuracy: {accuracy:.4f}")
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
