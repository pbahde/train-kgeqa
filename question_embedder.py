import torch
from transformers import AutoTokenizer, AutoModel


class QuestionEmbedder:
    def __init__(self, tokenizer_name: str, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_question(self, question: str) -> torch.Tensor:
        with torch.no_grad():
            # Tokenisierung der Frage
            inputs = self.tokenizer(question, return_tensors="pt")
            # Ausführen der Frage durch das Modell
            outputs = self.model(**inputs)
            # Extrahieren des finalen Hidden States (CLS-Token)
            embeddings = outputs.last_hidden_state[:, 0, :][0]
            return embeddings
