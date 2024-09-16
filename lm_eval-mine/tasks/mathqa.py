"""
MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms
https://arxiv.org/pdf/1905.13319.pdf

MathQA is a large-scale dataset of 37k English multiple-choice math word problems
covering multiple math domain categories by modeling operation programs corresponding
to word problems in the AQuA dataset (Ling et al., 2017).

Homepage: https://math-qa.github.io/math-QA/
"""
import re
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@misc{amini2019mathqa,
    title={MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms},
    author={Aida Amini and Saadia Gabriel and Peter Lin and Rik Koncel-Kedziorski and Yejin Choi and Hannaneh Hajishirzi},
    year={2019},
    eprint={1905.13319},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


class MathQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "math_qa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        answer_idx = ["a", "b", "c", "d", "e"].index(doc["correct"])
        choices = [
            c[4:].rstrip(" ,")
            for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", doc["options"])
        ]

        out_doc = {
            "query": "Question: " + doc["Problem"] + "\nAnswer:",
            "choices": choices,
            "gold": answer_idx,
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
