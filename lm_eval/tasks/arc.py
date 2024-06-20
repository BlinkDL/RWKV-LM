"""
Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
https://arxiv.org/pdf/1803.05457.pdf

The ARC dataset consists of 7,787 science exam questions drawn from a variety
of sources, including science questions provided under license by a research
partner affiliated with AI2. These are text-only, English language exam questions
that span several grade levels as indicated in the files. Each question has a
multiple choice structure (typically 4 answer options). The questions are sorted
into a Challenge Set of 2,590 “hard” questions (those that both a retrieval and
a co-occurrence method fail to answer correctly) and an Easy Set of 5,197 questions.

Homepage: https://allenai.org/data/arc
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@article{Clark2018ThinkYH,
  title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge},
  author={Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
  journal={ArXiv},
  year={2018},
  volume={abs/1803.05457}
}
"""


class ARCEasy(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Easy"

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
        # NOTE: Some `doc["answerKey"]`s are in numeric string format being one
        # of {'1', '2', '3', '4', '5'}. We map them back to letters.
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
        out_doc = {
            "id": doc["id"],
            "query": "Question: " + doc["question"] + "\nAnswer:",
            "choices": doc["choices"]["text"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class ARCChallenge(ARCEasy):
    DATASET_PATH = "ai2_arc"
    DATASET_NAME = "ARC-Challenge"
