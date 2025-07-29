import torch
import numpy as np
from .evaluator import Evaluator


class Fense:

    def __init__(self,
                 sbert_model="paraphrase-TinyBERT-L6-v2",
                 echecker_model="echecker_clotho_audiocaps_base",
                 penalty=0.9) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.evaluator = Evaluator(device=device, sbert_model=sbert_model,
            echecker_model=echecker_model, penalty=penalty)
        
    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        keys = list(gts.keys())
        list_cand = [res[key][0] for key in keys]
        list_refs = [gts[key] for key in keys]
        scores = self.evaluator.corpus_score(list_cand, list_refs, agg_score="none")
        average_score = np.mean(np.array(scores))
        return average_score, np.array(scores)

    def method(self):
        return "Fense"
