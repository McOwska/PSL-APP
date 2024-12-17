import torch
import time
from collections import deque


class GestureRecognitionHandler:
    def __init__(self, model, label_map, confidence_threshold=0.8, consecutive_frames=1, max_sequence_length=20):
        self.model = model
        self.actions = {value: key for key, value in label_map.items()}
        self.confidence_threshold = confidence_threshold
        self.consecutive_frames = consecutive_frames
        self.max_sequence_length = max_sequence_length

        self.buffer = deque(maxlen=max_sequence_length)
        self.confidence_window = []
        self.action_window = []
        self.hands_visible_window = []

        self.hand_left_indices = [idx for idx in range(91, 112)]
        self.hand_right_indices = [idx for idx in range(112, 133)]

        self.last_action = ""
        self.last_action_time = time.time()
        self.action_display_duration = 1

        self.flag = True

    def process_frame(self, frame, transform):
        """
        Przetwarza pojedynczą klatkę, aktualizuje bufor i dokonuje predykcji, gdy bufor osiągnie długość max_sequence_length.

        Args:
            frame: Pojedyncza klatka wejściowa (np. punkty kluczowe w kształcie [133, 2]).
            transform: Funkcja przekształcająca klatkę na tensor.

        Returns:
            Tuple[str, float]: Rozpoznana akcja i jej pewność lub (None, None), jeśli predykcja nie została wykonana.
        """
        # Dodajemy klatkę do bufora
        x = transform([frame])  # Przekształcenie klatki
        x = torch.tensor(x[0], dtype=torch.float32).view(-1)  # Spłaszczenie do [266]
        self.buffer.append(x)  # Dodanie klatki do bufora

        # Sprawdzenie widoczności dłoni
        hands_visible = (
            (frame[self.hand_left_indices, 0] != 0).any() or
            (frame[self.hand_left_indices, 1] != 0).any() or
            (frame[self.hand_right_indices, 0] != 0).any() or
            (frame[self.hand_right_indices, 1] != 0).any()
        )
        self.hands_visible_window.append(hands_visible)

        # Usuń nadmiarowe elementy w oknie widoczności dłoni
        if len(self.hands_visible_window) > 5:
            self.hands_visible_window.pop(0)

        # Reset bufora, jeśli brak widoczności dłoni przez 5 klatek
        if len(self.hands_visible_window) == 5 and not any(self.hands_visible_window):
            self.reset_buffer()
            return None, None

        # Dokonujemy predykcji, gdy bufor osiągnie maksymalną długość
        if len(self.buffer) == self.max_sequence_length:
            sequence = torch.stack(list(self.buffer)).unsqueeze(0)  # [1, seq_length, features]
            sequence = sequence.to(next(self.model.parameters()).device)

            self.model.eval()
            with torch.no_grad():
                outputs, _ = self.model(sequence)
                outputs = torch.nn.functional.softmax(outputs[0], dim=0)  # Softmax na wynikach
                confidence, predicted_index = torch.max(outputs, dim=0)
                
                predicted_action = self.actions[predicted_index.item()]

                print('Predicted action:', predicted_action, 'with confidence:', confidence.item())

            # Nie resetujemy bufora po predykcji, bo chcemy mieć ciągłą predykcję
            # Bufor jest maxlen=20 i się przesuwa z każdą nową klatką
            # Jeśli chciałbyś reset po każdej predykcji, możesz wywołać self.reset_buffer()
            if confidence.item() >= self.confidence_threshold:
                return predicted_action, confidence.item()
            # return predicted_action, confidence.item()

        return None, None

    def reset_buffer(self):
        """Resetuje bufor i związane okna."""
        self.buffer.clear()
        self.confidence_window.clear()
        self.action_window.clear()
        self.hands_visible_window.clear()
