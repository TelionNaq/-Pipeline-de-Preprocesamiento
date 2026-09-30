import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer

class ModernTokenizer:
    def __init__(self, model_name: str = "dccuchile/bert-base-spanish-wwm-cased"):
        """
        Inicializa el tokenizador basado en subwords (Subword / BPE / WordPiece).
        Por defecto utilizamos BETO (BERT preentrenado en Español).
        """
        print(f"Cargando tokenizador moderno: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def process_texts(self, texts: list[str], max_length: int = 32) -> dict:
        """
        Tokeniza y codifica una lista de textos.
        Devuelve diccionarios con tensores (input_ids y attention_mask).
        """
        encoded_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Extraer los tokens decodificados para inspección visual
        tokens_list = [self.tokenizer.convert_ids_to_tokens(ids) for ids in encoded_dict["input_ids"]]
        
        return {
            "input_ids": encoded_dict["input_ids"],
            "attention_mask": encoded_dict["attention_mask"],
            "tokens": tokens_list
        }
