import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

from question_embedder import QuestionEmbedder

# Configure logging
logging.basicConfig(filename='logs.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuestionNetwork(nn.Module):
    def __init__(self, kge_model, kge_model_type):
        super(QuestionNetwork, self).__init__()
        self.kge_model = kge_model
        self.kge_model_type = kge_model_type
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 100),
        )

    def forward(self, question_emb):
        rel_emb = self.linear_relu_stack(question_emb)
        return rel_emb

    def train_model(self, trainData, testData, numEpos):
        self.parameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for epoch in range(0, numEpos):
            logging.info(f"EPOCH: {epoch}")
            self._training_step(optimizer, trainData)
            self._test(testData)

    def _training_step(self, optimizer, trainData):
        dataLoader = DataLoader(trainData, batch_size=256, shuffle=False)
        logging.info("Train")
        for e, q, t in dataLoader:
            logging.info("Batch")
            optimizer.zero_grad()
            outputs = self(q)
            loss = self.custom_loss_function(outputs, e, t)
            loss.backward()
            optimizer.step()

    def _test(self, testData):
        logging.info("test")
        self.eval()
        dataLoader = DataLoader(testData, batch_size=256, shuffle=False)
        loss = 0
        cnt = 0
        for e, q, t in dataLoader:
            logging.info("testbatch")
            outputs = self(q)
            loss += self.custom_loss_function(outputs, e, t)
            cnt += 1
        logging.info(f"AVG Loss: {loss / cnt}")

    def valid(self, valid_data):
        dataLoader = DataLoader(valid_data, batch_size=256, shuffle=False)
        loss = 0
        cnt = 0
        logging.info("VALID")
        for e, q, t in dataLoader:
            logging.info("testbatch")
            outputs = self(q)
            loss += self.custom_loss_function(outputs, e, t)
            cnt += 1
        logging.info(f"AVG Loss: {loss / cnt}")

    def custom_loss_function(self, output, topic_entity, target):
        scores = self.kge_model.score(self.kge_model_type, topic_entity, output)

        output_sigmoid = nn.Sigmoid()(scores)
        loss_function = nn.BCELoss()
        loss = loss_function(output_sigmoid, target)

        return loss
