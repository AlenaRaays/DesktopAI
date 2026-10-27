import numpy as np

class TinyNumPyAgent:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        np.random.seed(42)
        # Исправлено: 2.0 вместо 2,0 и убран .astype(np.ndarray) у скаляра
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:  # Вынесли на уровень класса (исправили отступ)
        return np.maximum(0, x)
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:  # Исправили опечатку _sortmax -> _softmax
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x: np.ndarray) -> np.ndarray:  # Вынесли на уровень класса
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self._relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2

        probabilities = self._softmax(self.z2)
        return probabilities
    
    def predict_action(self, state_vector: np.ndarray) -> int:  # Вынесли на уровень класса
        probs = self.forward(state_vector)
        return int(np.argmax(probs))
