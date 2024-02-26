import torch

from qa_data_set import QADataset
from question_embedder import QuestionEmbedder
from question_nn import QuestionNetwork

#initilizeKGEModel
from kg_embedding_model import KGEmbeddingModel
model = KGEmbeddingModel.load_from_dir("data/embeddings/nations")
#initilize network
net = QuestionNetwork(model, "complex")


train_dataSet = QADataset("data/embeddings/nations/train.txt", model)
test_dataSet = QADataset("data/embeddings/nations/test.txt", model)
valid_dataSet = QADataset("data/embeddings/nations/valid.txt", model)

net.train_model(train_dataSet, test_dataSet, 10)
net.valid(valid_dataSet)
torch.save(net, 'model.pth')



