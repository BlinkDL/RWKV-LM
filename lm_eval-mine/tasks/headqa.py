"""
Interpretable Multi-Step Reasoning with Knowledge Extraction on Complex Healthcare Question Answering
https://aclanthology.org/P19-1092.pdf

HEAD-QA is a multi-choice HEAlthcare Dataset. The questions come from exams to
access a specialized position in the Spanish healthcare system, and are challenging
even for highly specialized humans.

Homepage: https://aghie.github.io/head-qa/
"""
import inspect
import lm_eval.datasets.headqa.headqa
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@misc{liu2020interpretable,
    title={Interpretable Multi-Step Reasoning with Knowledge Extraction on Complex Healthcare Question Answering},
    author={Ye Liu and Shaika Chowdhury and Chenwei Zhang and Cornelia Caragea and Philip S. Yu},
    year={2020},
    eprint={2008.02434},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
"""


class HeadQABase(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = inspect.getfile(lm_eval.datasets.headqa.headqa)

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
        out_doc = {
            "id": doc["qid"],
            "query": "Question: " + doc["qtext"] + "\nAnswer:",
            "choices": [answer["atext"] for answer in doc["answers"]],
            "gold": int(doc["ra"]) - 1,
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class HeadQAEn(HeadQABase):
    DATASET_NAME = "en"


class HeadQAEs(HeadQABase):
    DATASET_NAME = "es"


# for backwards compatibility
class HeadQAEsDeprecated(HeadQABase):
    DATASET_NAME = "es"

    def __init__(self):
        super().__init__()
        print(
            "WARNING: headqa is deprecated. Please use headqa_es or headqa_en instead. See https://github.com/EleutherAI/lm-evaluation-harness/pull/240 for more info."
        )
