"""
PROST: Physical Reasoning about Objects Through Space and Time
https://arxiv.org/pdf/2106.03634.pdf

PROST, Physical Reasoning about Objects Through Space and Time, is a dataset
consisting of 18,736 multiple-choice questions made from 14 manually curated
templates, covering 10 physical reasoning concepts. All questions are designed
to probe both causal and masked language models in a zero-shot setting.

NOTE: PROST is limited to the zero-shot setting to adhere to authors' intentions
as discussed in section 7 of the paper: "We hope that the community will use
this dataset in the intended way: in a zero-shot setting to probe models which
have been trained on data not specifically collected to succeed on PROST."

Homepage: https://github.com/nala-cub/prost
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{aroca-ouellette-etal-2021-prost,
    title = "{PROST}: {P}hysical Reasoning about Objects through Space and Time",
    author = "Aroca-Ouellette, St{\'e}phane  and
      Paik, Cory  and
      Roncone, Alessandro  and
      Kann, Katharina",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.404",
    pages = "4597--4608",
}
"""


class PROST(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "corypaik/prost"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        assert (
            num_fewshot == 0
        ), "PROST is designed to probe models in a zero-shot fashion only."
        return super().fewshot_context(
            doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
        )

    def _process_doc(self, doc):
        out_doc = {
            "query": f"{doc['context']}\nQuestion: {doc['ex_question']}\nAnswer:",
            "choices": [doc["A"], doc["B"], doc["C"], doc["D"]],
            "gold": doc["label"],
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]
