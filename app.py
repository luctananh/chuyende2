import tensorflow.keras as keras
import sys
sys.modules['keras'] = keras

import streamlit as st
import torch
import numpy as np
from joblib import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Tải các tài nguyên cần thiết của NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Hàm tiền xử lý văn bản
def preprocess_text(text, use_lemmatization=True):
    text = text.lower()  # Chuyển về chữ thường
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Hàm tải model và các đối tượng liên quan
@st.cache_resource
def load_sentiment_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # Tải tokenizer, label encoder và mô hình Word2Vec đã lưu
        tokenizer, label_encoder, word2vec_model = load("models/tokenizer_and_encoder.joblib")
        print(type(tokenizer))
    except Exception as e:
        st.error(f"Lỗi tải tokenizer và encoder: {e}")
        return None, None, None, None, None

    # Xây dựng embedding matrix dựa trên mô hình Word2Vec
    vocab_size = len(word2vec_model.wv.key_to_index) + 1
    embedding_dim = 100
    lstm_units = 128
    max_len = 50
    num_classes = len(label_encoder.classes_)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i, word in enumerate(word2vec_model.wv.key_to_index):
        embedding_matrix[i + 1] = word2vec_model.wv[word]

    # Xây dựng kiến trúc của mô hình BiLSTM
    import torch.nn as nn
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

    model = BiLSTMModel(vocab_size, embedding_dim, lstm_units, num_classes, embedding_matrix, max_len)
    try:
        model.load_state_dict(torch.load("models/BiLSTM_Word2Vec_model.pt", map_location=device))
    except Exception as e:
        st.error(f"Lỗi tải model: {e}")
        return None, None, None, None, None
    model = model.to(device)
    return model, tokenizer, label_encoder, device, max_len

# Tải model khi ứng dụng khởi chạy
model, tokenizer, label_encoder, device, max_len = load_sentiment_model()
if model is None:
    st.error("Không tải được model, vui lòng kiểm tra lại các file đã lưu.")

# Giao diện của ứng dụng
st.title("Sentiment Analysis Interface")
st.write("Nhập văn bản vào ô dưới đây và nhấn **Predict** để dự đoán cảm xúc của văn bản.")

# Text area để người dùng nhập văn bản
input_text = st.text_area("Input Text", "This is an amazing movie! I really enjoyed it.")

# Khi người dùng nhấn nút Predict
if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Vui lòng nhập văn bản.")
    else:
        # Tiền xử lý văn bản
        processed_text = preprocess_text(input_text)
        # Chuyển văn bản thành chuỗi số thông qua tokenizer và padding
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        input_tensor = torch.tensor(padded_sequence, dtype=torch.long).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            _, prediction = torch.max(outputs, 1)
            prediction = prediction.cpu().numpy()[0]
        # Giải mã nhãn dự đoán
        predicted_sentiment = label_encoder.inverse_transform([prediction])[0]
        st.write("Predicted Sentiment:", predicted_sentiment)
