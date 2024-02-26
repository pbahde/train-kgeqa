import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

class KGEmbeddingModel():
    def __init__(self,
                 relation_embeddings_path, entity_embeddings_path,
                 relation_to_id_path, entity_to_id_path,
                 id_to_relation_path, id_to_entity_path):
        self._relation_embeddings = self._load_embedding(relation_embeddings_path)
        self._entity_embeddings = self._load_embedding(entity_embeddings_path)
        self._relation_to_id = self._load_dict(relation_to_id_path)
        self._entity_to_id = self._load_dict(entity_to_id_path)
        self._id_to_relation = self._load_dict(id_to_relation_path)
        self._id_to_entity = self._load_dict(id_to_entity_path)

    @staticmethod
    def load_from_dir(src_dir: str):
        model = KGEmbeddingModel(src_dir+"/relation_embeddings.npy",
                                 src_dir+"/entitiy_embeddings.npy",
                                 src_dir+"/relation_to_id.pkl",
                                 src_dir+"/entity_to_id.pkl",
                                 src_dir+"/id_to_relation.pkl",
                                 src_dir+"/id_to_entity.pkl")
        return model
    def _load_embedding(self, path) -> Tensor:
        numpy_embeddings = np.load(path)
        tensor_embeddings = torch.from_numpy(numpy_embeddings)
        tensor_embeddings_list = [Tensor(a) for a in tensor_embeddings]
        return torch.stack(tensor_embeddings_list, dim=0)

    def _load_dict(self, path) -> dict:
        with open(path, 'rb') as f:
            loaded_dict = pickle.load(f)
            return loaded_dict

    def _score_complex(self, s_emb: Tensor, r_emb: Tensor, o_emb: Tensor):
        n = r_emb.size(0)

        # Split the relation and object embeddings into real part (first half) and
        # imaginary part (second half).
        p_emb_re, p_emb_im = (t.contiguous() for t in r_emb.chunk(2, dim=1))
        o_emb_re, o_emb_im = (t.contiguous() for t in o_emb.chunk(2, dim=1))
        # combine them again to create a column block for each required combination
        s_all = torch.cat((s_emb, s_emb), dim=1)  # re, im, re, im
        r_all = torch.cat((p_emb_re, r_emb, -p_emb_im), dim=1)  # re, re, im, -im
        o_all = torch.cat((o_emb, o_emb_im, o_emb_re), dim=1)  # re, im, im, re

        out = (s_all * r_all).mm(o_all.transpose(0, 1))
        return out.view(n, -1)
    def _score_transe(self, s_emb: Tensor, r_emb: Tensor, o_emb: Tensor):
        n = r_emb.size(0)
        # we do not use matrix multiplication due to this issue
        # https://github.com/pytorch/pytorch/issues/42479
        out = torch.cdist(
            s_emb + r_emb,
            o_emb,
            p=2,
            compute_mode="donot_use_mm_for_euclid_dist",
        )
        return out.view(n, -1)

    @staticmethod
    def _transfer(ent_emb, norm_vec_emb):
        norm_vec_emb = F.normalize(norm_vec_emb, p=2, dim=-1)
        return (
                ent_emb
                - torch.sum(ent_emb * norm_vec_emb, dim=-1, keepdim=True) * norm_vec_emb
        )
    def _score_transh(self, s_emb: Tensor, r_emb: Tensor, o_emb: Tensor):
        # split relation embeddings into "rel_emb" and "norm_vec_emb"
        rel_emb, norm_vec_emb = torch.chunk(r_emb, 2, dim=1)
        # projected once for every different relation p. Unclear if this can be avoided.

        n = r_emb.size(0)
        # n = n_s = n_p != n_o = m
        m = o_emb.shape[0]
        s_translated = self._transfer(s_emb, norm_vec_emb) + rel_emb
        s_translated = s_translated.repeat(m, 1)
        # s_translated has shape [(m * n), dim]
        o_emb = o_emb.unsqueeze(1)
        o_emb = o_emb.repeat(1, n, 1)
        # o_emb has shape [m, n, dim]
        # norm_vec_emb has shape [n, dim]
        # --> make use of broadcasting semantics
        o_emb = self._transfer(o_emb, norm_vec_emb)
        o_emb = o_emb.view(-1, o_emb.shape[-1])
        # o_emb has shape [(m * n), dim]
        # --> perform pairwise distance calculation
        out = -F.pairwise_distance(s_translated, o_emb, p=1)
        # out has shape [(m * n)]
        # --> transform shape to [n, m]
        out = out.view(m, n)
        out = out.transpose(0, 1)

        return -out.view(n, -1)
    def _score_cosine(self, s_emb: Tensor, r_emb: Tensor, o_emb: Tensor):
        n = r_emb.size(0)
        # Berechne die Summe von s_emb und r_emb
        sum_emb = s_emb + r_emb

        # Berechne die Cosinus-Ähnlichkeit zwischen sum_emb und o_emb
        out = F.cosine_similarity(sum_emb, o_emb, dim=-1)
        return out.view(n, -1)
    def score(self, scoring_function, subject, relation, objects = None):
        s_emb = self.ensure_2d_tensor(self.convert_entity_to_tensor(subject))
        p_emb = self.ensure_2d_tensor(self.convert_relation_to_tensor(relation))
        if(objects == None):
            objects = self._entity_embeddings
        o_emb = self.ensure_2d_tensor(objects if isinstance(objects, Tensor) else self.convert_entity_list_to_tensor(objects))

        if scoring_function == "complex":
            return self._score_complex(s_emb, p_emb, o_emb)
        elif scoring_function == "transe":
            return self._score_transe(s_emb, p_emb, o_emb)
        elif scoring_function == "transh":
            return self._score_transh(s_emb, p_emb, o_emb)
        elif scoring_function == "cosine":
            return self._score_cosine(s_emb, p_emb, o_emb)
        else:
            raise ValueError("Invalid scoring function. Must be 'complex' or 'transe'")

    def ensure_2d_tensor(self, tensor: Tensor):
        if tensor.dim() == 1:  # Wenn der Tensor eine Dimension hat
            return tensor.unsqueeze(0)  # Hinzufügen einer zusätzlichen Dimension am Anfang
        else:
            return tensor

    def convert_entity_to_tensor(self, input) -> Tensor:
        if isinstance(input, Tensor):
            return input
        elif isinstance(input, int):
            return self._entity_embeddings[input]
        elif isinstance(input, str):
            return self._entity_embeddings[self._entity_to_id[input]]
        else:
            raise ValueError("Invalid input Subjects / Objects must be int, string or tensor")
    def convert_relation_to_tensor(self, input) -> Tensor:
        if isinstance(input, Tensor):
            return input
        elif isinstance(input, int):
            return self._relation_embeddings[input]
        elif isinstance(input, str):
            return self._relation_embeddings[self._relation_to_id[input]]
        else:
            raise ValueError("Invalid input relation must be int, string or tensor")
    def convert_entity_list_to_tensor(self, objects):
        if isinstance(objects, list):
            if all(isinstance(item, type(objects[0])) for item in objects):
                tensor_list = []
                for item in objects:
                    tensor_list.append(self.convert_entity_to_tensor(item))
                return torch.stack(tensor_list, dim=0)
            else:
                raise ValueError("All elements in objects list must have the same type")
        else:
            raise ValueError("Objects must be a list or tensor")
    def get_top_k_scores(self, scoring_function: str, subj: str, pred, k: int) -> (float,str):
        result = self.score(scoring_function, subj, pred, self._entity_embeddings)
        largest = True if scoring_function in ["complex","cosine"] else False
        values, indices = torch.topk(result[0], k, largest=largest)
        return [(values[i].item(), self._id_to_entity[indices[i].item()]) for i in range(len(values))]


