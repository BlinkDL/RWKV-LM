"""
MuTual: A Dataset for Multi-Turn Dialogue Reasoning
https://www.aclweb.org/anthology/2020.acl-main.130/

MuTual is a retrieval-based dataset for multi-turn dialogue reasoning, which is
modified from Chinese high school English listening comprehension test data.

Homepage: https://github.com/Nealcly/MuTual
"""
import numpy as np
import inspect
import lm_eval.datasets.mutual.mutual
from lm_eval.base import Task, rf
from lm_eval.metrics import mean


_CITATION = """
@inproceedings{mutual,
    title = "MuTual: A Dataset for Multi-Turn Dialogue Reasoning",
    author = "Cui, Leyang  and Wu, Yu and Liu, Shujie and Zhang, Yue and Zhou, Ming" ,
    booktitle = "Proceedings of the 58th Conference of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
"""


class MuTualBase(Task):
    VERSION = 1
    DATASET_PATH = inspect.getfile(lm_eval.datasets.mutual.mutual)
    DATASET_NAME = None
    CHOICES = ["A", "B", "C", "D"]

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return NotImplemented

    def doc_to_text(self, doc):
        return self.detokenize(doc["article"])

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["article"]

    def doc_to_target(self, doc):
        return " " + self.detokenize(doc["options"][self.CHOICES.index(doc["answers"])])

    def construct_requests(self, doc, ctx):
        lls = []
        for option in doc["options"]:
            lls.append(rf.loglikelihood(ctx, f" {self.detokenize(option)}")[0])
        return lls

    def detokenize(self, text):
        text = text.replace(" '", "'")
        text = text.replace(" \n", "\n")
        text = text.replace("\n ", "\n")
        text = text.replace(" n't", "n't")
        text = text.replace("`` ", '"')
        text = text.replace("''", '"')
        # punctuation
        text = text.replace(" :", ":")
        text = text.replace(" ;", ";")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        return text

    def process_results(self, doc, results):
        gold = self.CHOICES.index(doc["answers"])
        r4_1 = np.argmax(results) == gold  # r4_1 = accuracy
        ranks = sorted(results, reverse=True)
        r4_2 = (ranks.index(results[gold]) == 1) + r4_1
        mrr = 1.0 / (ranks.index(results[gold]) + 1)  # `+ 1` for index offset
        return {"r@1": r4_1, "r@2": r4_2, "mrr": mrr}

    def aggregation(self):
        return {"r@1": mean, "r@2": mean, "mrr": mean}

    def higher_is_better(self):
        return {"r@1": True, "r@2": True, "mrr": True}


class MuTual(MuTualBase):
    DATASET_NAME = "mutual"


class MuTualPlus(MuTualBase):
    DATASET_NAME = "mutual_plus"
