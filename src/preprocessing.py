import re
import nltk
from collections import Counter

# Asegurar la descarga de recursos de nltk requeridos
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self, language='spanish'):
        self.stop_words = set(stopwords.words(language))
        
    def clean_text(self, text: str) -> str:
        """
        Limpia y normaliza el texto.
        - Convierte a minúsculas.
        - Elimina URLs.
        - Elimina caracteres especiales innecesarios pero mantiene signos de puntuación clave.
        """
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Reemplazar múltiples espacios por uno solo
        text = re.sub(r'\s+', ' ', text)
        # Conservar letras, números, signos de puntuación básicos y tildes
        text = re.sub(r'[^a-záéíóúñü0-9\s.,!?#¡¿]', '', text)
        return text.strip()

    def remove_stopwords(self, text: str) -> str:
        """
        Elimina las stopwords del texto. Útil para enfoques clásicos de BoW o TF-IDF.
        Nota: No siempre es recomendado para modelos de Deep Learning tipo Transformer, 
        ya que necesitan el contexto gramatical completo.
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return " ".join(filtered_words)


class ClassicalTokenizer:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_built = False

    def build_vocab(self, texts: list[str]):
        """Construye el vocabulario léxico a partir de una lista de textos."""
        word_counts = Counter()
        for text in texts:
            # Tokenización básica mediante expresiones regulares
            words = re.findall(r'\b\w+\b|[.,!?#¡¿]', text)
            word_counts.update(words)
        
        # Seleccionar las palabras más comunes
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        self.vocab_built = True

    def tokenize_and_encode(self, text: str, max_length: int = 20) -> list[int]:
        """Tokeniza el texto y lo convierte a secuencias de IDs numéricos con padding."""
        if not self.vocab_built:
            raise ValueError("El vocabulario no ha sido construido. Llama a build_vocab primero.")
            
        words = re.findall(r'\b\w+\b|[.,!?#¡¿]', text)
        encoded = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
        
        # Truncamiento (si la secuencia es mayor a max_length)
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        
        # Padding (relleno para secuencias más cortas)
        padding_length = max_length - len(encoded)
        encoded = encoded + [self.word2idx["<PAD>"]] * padding_length
        
        return encoded
