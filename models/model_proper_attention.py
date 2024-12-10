import torch
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_length, hidden_size]
        attention_scores = self.attention(lstm_output)  # [batch_size, seq_length, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, seq_length, 1]
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # [batch_size, hidden_size]
        return context_vector, attention_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.1):
        super(LSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: [batch_size, seq_length, input_size]
        x = self.prepare_input(x)  # [batch_size, seq_length, normalized_features]
        # print(f"Input shape after normalization: {x.shape}")
        lstm_output, _ = self.lstm(x)  # lstm_output: [batch_size, seq_length, hidden_size]
        context_vector, attention_weights = self.attention(lstm_output)  # [batch_size, hidden_size], [batch_size, seq_length, 1]
        output = self.fc(context_vector)  # [batch_size, num_classes]
        return output, attention_weights

    def prepare_input(self, x: torch.Tensor):
        """
        Normalizacja dla całej sekwencji.
        x: Tensor o kształcie [batch_size, seq_length, input_features] (input_features = num_points * coords)
        Zwraca tensor o kształcie [batch_size, seq_length, normalized_features]
        """
        batch_size, seq_length, input_features = x.shape
        num_points = 133  # Liczba punktów w każdej klatce
        coords = 2  # Liczba współrzędnych na punkt (x, y)

        # Przekształcenie do [batch_size, seq_length, num_points, coords]
        x = x.view(batch_size, seq_length, num_points, coords)

        normalized_sequences = []

        for batch_idx in range(batch_size):
            sequence = x[batch_idx]  # [seq_length, num_points, coords]
            normalized_sequence = []

            for frame in sequence:  # Przechodzimy przez każdą klatkę
                body = frame[:17]
                left = frame[91:112]
                right = frame[112:]
                source_body = frame[0].clone()
                source_left = frame[100].clone()
                source_right = frame[121].clone()

                # Obliczanie szerokości i wysokości
                w_left = torch.max(left[:, 0]) - torch.min(left[:, 0])
                h_left = torch.max(left[:, 1]) - torch.min(left[:, 1])
                w_right = torch.max(right[:, 0]) - torch.min(right[:, 0])
                h_right = torch.max(right[:, 1]) - torch.min(right[:, 1])
                w_body = body[5][0] - body[6][0]
                h_body = 4 * w_body

                # Normalizacja
                face2left = left - source_body
                face2right = right - source_body
                hand2hand = left - right
                left -= source_left
                right -= source_right

                if w_left != 0. and h_left != 0:
                    left[:, 0] /= w_left
                    left[:, 1] /= h_left

                if w_right != 0. and h_right != 0:
                    right[:, 0] /= w_right
                    right[:, 1] /= h_right

                if w_body != 0. and h_body != 0:
                    face2left[:, 0] /= w_body
                    face2right[:, 1] /= h_body
                    hand2hand[:, 0] /= w_body
                    hand2hand[:, 1] /= h_body

                # Łączenie wszystkich cech
                normalized_frame = torch.cat([left, right, face2left, face2right, hand2hand])
                normalized_sequence.append(normalized_frame.view(-1))  # Spłaszczenie ramki

            normalized_sequences.append(torch.stack(normalized_sequence))  # [seq_length, normalized_features]

        return torch.stack(normalized_sequences)  # [batch_size, seq_length, normalized_features]


