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

class LSTMs4WithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.0):
        super(LSTMs4WithAttention, self).__init__()
        self.lstm_right_hand = nn.LSTM(42, 64, batch_first=True)
        self.lstm_right_arm = nn.LSTM(6, 32, batch_first=True)
        self.lstm_left_hand = nn.LSTM(42, 64, batch_first=True)
        self.lstm_left_arm = nn.LSTM(6, 32, batch_first=True)
        self.attention = AttentionLayer(2*64+2*32)
        self.fc = nn.Linear(2*64+2*32, num_classes)
    
    def forward(self, x):
        # x: [batch_size, seq_length, normalized_features]
        # Najpierw dzielimy dane na cztery grupy
        right_hand_input, right_arm_input, left_hand_input, left_arm_input = self.prepare_input(x)
        
        # Przepuszczamy każdą grupę przez jej LSTM
        right_hand_output, _ = self.lstm_right_hand(right_hand_input)   # [batch_size, seq_length, hidden_size]
        right_arm_output, _ = self.lstm_right_arm(right_arm_input)      # [batch_size, seq_length, hidden_size]
        left_hand_output, _ = self.lstm_left_hand(left_hand_input)      # [batch_size, seq_length, hidden_size]
        left_arm_output, _ = self.lstm_left_arm(left_arm_input)         # [batch_size, seq_length, hidden_size]
        
        # Łączymy cztery wyjścia wzdłuż wymiaru cech
        combined_output = torch.cat([right_hand_output, right_arm_output, left_hand_output, left_arm_output], dim=2)
        # combined_output: [batch_size, seq_length, 4*hidden_size]
        
        # Przykładamy uwagę
        context_vector, attention_weights = self.attention(combined_output)
        # context_vector: [batch_size, 4*hidden_size]
        
        # Klasyfikacja
        output = self.fc(context_vector)  # [batch_size, num_classes]
        
        return output, attention_weights

    def prepare_input(self, x: torch.Tensor):
        """
        Normalizacja dla całej sekwencji.
        x: Tensor o kształcie [batch_size, seq_length, input_features] (input_features = num_points * coords)
        Zwraca 4 tensory:
        - right_hand_input: [batch_size, seq_length, features_right_hand]
        - right_arm_input: [batch_size, seq_length, features_right_arm]
        - left_hand_input: [batch_size, seq_length, features_left_hand]
        - left_arm_input: [batch_size, seq_length, features_left_arm]
        """
        batch_size, seq_length, input_features = x.shape
        num_points = 133  # Liczba punktów w każdej klatce
        coords = 2  # Liczba współrzędnych na punkt (x, y)

        # Przekształcenie do [batch_size, seq_length, num_points, coords]
        x = x.view(batch_size, seq_length, num_points, coords)

        # Listy do przechowywania wyników dla całego batcha
        all_left_hand = []
        all_right_hand = []
        all_left_arm = []
        all_right_arm = []

        for batch_idx in range(batch_size):
            sequence = x[batch_idx]  # [seq_length, num_points, coords]

            # Listy do przechowywania wyników dla pojedynczej sekwencji (pojedynczy batch_idx)
            normalized_left_hand_seq = []
            normalized_right_hand_seq = []
            normalized_left_arm_seq = []
            normalized_right_arm_seq = []

            for frame in sequence:  # Przechodzimy przez każdą klatkę
                # Wyodrębnianie danych
                body = frame[:17]
                left = frame[91:112]     # punkty lewej dłoni i/lub ręki (zależnie od indeksów)
                right = frame[112:]      # punkty prawej dłoni i/lub ręki
                left_wrist = frame[9]
                right_wrist = frame[10]
                left_arm = torch.stack([frame[5], frame[7], frame[9]])    # kilka punktów lewej ręki
                right_arm = torch.stack([frame[6], frame[8], frame[10]])  # kilka punktów prawej ręki

                # Wybór środka ciała jako punktu odniesienia
                center_body = body[0].clone()  # Przyjmujemy pierwszy punkt jako środek ciała

                # Przesunięcie rąk względem pozycji nadgarstków
                left_relative = left - left_wrist
                right_relative = right - right_wrist

                # Obliczanie szerokości i wysokości dla lewej dłoni
                w_left = torch.max(left_relative[:, 0]) - torch.min(left_relative[:, 0])
                h_left = torch.max(left_relative[:, 1]) - torch.min(left_relative[:, 1])
                # Obliczanie szerokości i wysokości dla prawej dłoni
                w_right = torch.max(right_relative[:, 0]) - torch.min(right_relative[:, 0])
                h_right = torch.max(right_relative[:, 1]) - torch.min(right_relative[:, 1])

                # Skala dla lewej i prawej dłoni
                scale_left = torch.max(w_left, h_left)
                scale_right = torch.max(w_right, h_right)

                scale_left = scale_left if scale_left != 0 else 1.0
                scale_right = scale_right if scale_right != 0 else 1.0

                # Normalizacja dłoni
                left_relative[:, 0] /= scale_left
                left_relative[:, 1] /= scale_left

                right_relative[:, 0] /= scale_right
                right_relative[:, 1] /= scale_right

                # Ramiona względem środka ciała
                left_arm_relative = left_arm - center_body
                right_arm_relative = right_arm - center_body

                w_left_arm = torch.max(left_arm_relative[:, 0]) - torch.min(left_arm_relative[:, 0])
                h_left_arm = torch.max(left_arm_relative[:, 1]) - torch.min(left_arm_relative[:, 1])
                w_right_arm = torch.max(right_arm_relative[:, 0]) - torch.min(right_arm_relative[:, 0])
                h_right_arm = torch.max(right_arm_relative[:, 1]) - torch.min(right_arm_relative[:, 1])

                scale_left_arm = torch.max(w_left_arm, h_left_arm)
                scale_right_arm = torch.max(w_right_arm, h_right_arm)

                scale_left_arm = scale_left_arm if scale_left_arm != 0 else 1.0
                scale_right_arm = scale_right_arm if scale_right_arm != 0 else 1.0

                left_arm_relative /= scale_left_arm
                right_arm_relative /= scale_right_arm

                # Teraz zamiast łączyć wszystko w jeden wektor, 
                # rozdzielamy dane na cztery osobne części.
                # Każdą "klatkę" reprezentujemy oddzielnie dla lewej dłoni, prawej dłoni, lewego ramienia i prawego ramienia.
                
                # Spłaszczamy, bo LSTM oczekuje wejścia [batch, seq, features]
                left_hand_feats = left_relative.view(-1)
                right_hand_feats = right_relative.view(-1)
                left_arm_feats = left_arm_relative.view(-1)
                right_arm_feats = right_arm_relative.view(-1)

                # Dodajemy do list
                normalized_left_hand_seq.append(left_hand_feats)
                normalized_right_hand_seq.append(right_hand_feats)
                normalized_left_arm_seq.append(left_arm_feats)
                normalized_right_arm_seq.append(right_arm_feats)

            # Po przetworzeniu całej sekwencji (wszystkich klatek) dla jednego przykładu z batcha:
            normalized_left_hand_seq = torch.stack(normalized_left_hand_seq)     # [seq_length, feats_left_hand]
            normalized_right_hand_seq = torch.stack(normalized_right_hand_seq)   # [seq_length, feats_right_hand]
            normalized_left_arm_seq = torch.stack(normalized_left_arm_seq)       # [seq_length, feats_left_arm]
            normalized_right_arm_seq = torch.stack(normalized_right_arm_seq)     # [seq_length, feats_right_arm]

            # Dodajemy do list batchowych
            all_left_hand.append(normalized_left_hand_seq)
            all_right_hand.append(normalized_right_hand_seq)
            all_left_arm.append(normalized_left_arm_seq)
            all_right_arm.append(normalized_right_arm_seq)

        # Po przetworzeniu całego batcha:
        left_hand_batch = torch.stack(all_left_hand)    # [batch_size, seq_length, feats_left_hand]
        right_hand_batch = torch.stack(all_right_hand)  # [batch_size, seq_length, feats_right_hand]
        left_arm_batch = torch.stack(all_left_arm)      # [batch_size, seq_length, feats_left_arm]
        right_arm_batch = torch.stack(all_right_arm)    # [batch_size, seq_length, feats_right_arm]

        return right_hand_batch, right_arm_batch, left_hand_batch, left_arm_batch

