import os
import pickle
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from question_embedder import QuestionEmbedder


class QADataset(Dataset):
    def __init__(self, path, kge_model, tokenizer="bert-base-uncased", transformer_model="bert-base-uncased"):
        self.kge_model = kge_model
        self.question_embedder = QuestionEmbedder(tokenizer, transformer_model)
        self.entities, self.questions, self.answer_list = self.load_data(path)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.entities[idx], self.questions[idx], self.answer_list[idx]

    def load_data(self, file_path):
        entities = []
        questions = []
        answers_list = []

        with open(file_path, 'r') as file:
            for line in file:
                entity, question, *answers_combined = line.strip().split('\t')
                answers = [answer.strip() for answer in answers_combined[0].split(',')]
                entities.append(self.kge_model.convert_entity_to_tensor(entity))
                questions.append(self.question_embedder.embed_question(question))
                answers_indices = []
                for answer in answers:
                    answers_indices.append(self.kge_model._entity_to_id[answer])
                answers_list.append(self.get_target_tensor(answers_indices))
        return entities, questions, answers_list

    def get_target_tensor(self,indices):
        indices = torch.LongTensor(indices)
        vec_len = len(self.kge_model._entity_to_id)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot

