"""
Similarity of Semantic Relations
https://arxiv.org/pdf/cs/0608100.pdf

SAT (Scholastic Aptitude Test) Analogy Questions is a dataset comprising 374
multiple-choice analogy questions; 5 choices per question.

Homepage: https://aclweb.org/aclwiki/SAT_Analogy_Questions_(State_of_the_art)
"""
import inspect
import lm_eval.datasets.sat_analogies.sat_analogies
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@article{article,
    author = {Turney, Peter},
    year = {2006},
    month = {09},
    pages = {379-416},
    title = {Similarity of Semantic Relations},
    volume = {32},
    journal = {Computational Linguistics},
    doi = {10.1162/coli.2006.32.3.379}
}
"""


class SATAnalogies(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = inspect.getfile(lm_eval.datasets.sat_analogies.sat_analogies)
    DATASET_NAME = None

    def __init__(self, data_dir: str):
        """
        SAT Analog Questions is not publicly available. You must request the data
        by emailing Peter Turney and then download it to a local directory path
        which should be passed into the `data_dir` arg.
        """
        super().__init__(data_dir=data_dir)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return []

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return []

    def _process_doc(self, doc):
        return {
            "source": doc["source"],
            "query": doc["stem"].split(" ")[:2],
            "choices": [
                "{} is to {}".format(*c.split(" ")[:2]) for c in doc["choices"]
            ],
            "gold": ["a", "b", "c", "d", "e"].index(doc["solution"].strip()),
        }

    def doc_to_text(self, doc):
        return "{} is to {} as".format(*doc["query"])

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["source"] + "\n" + " ".join(doc["query"])
