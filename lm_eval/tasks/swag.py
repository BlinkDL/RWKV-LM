"""
SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference
https://arxiv.org/pdf/1808.05326.pdf

SWAG (Situations With Adversarial Generations) is an adversarial dataset
that consists of 113k multiple choice questions about grounded situations. Each
question is a video caption from LSMDC or ActivityNet Captions, with four answer
choices about what might happen next in the scene. The correct answer is the
(real) video caption for the next event in the video; the three incorrect
answers are adversarially generated and human verified, so as to fool machines
but not humans.

Homepage: https://rowanzellers.com/swag/
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{zellers2018swagaf,
    title={SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference},
    author={Zellers, Rowan and Bisk, Yonatan and Schwartz, Roy and Choi, Yejin},
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year={2018}
}
"""


class SWAG(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "swag"
    DATASET_NAME = "regular"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "query": doc["startphrase"],
            "choices": [doc["ending0"], doc["ending1"], doc["ending2"], doc["ending3"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]
