"""
QA4MRE 2011-2013: Overview of Question Answering for Machine Reading Evaluation
https://www.cs.cmu.edu/~./hovy/papers/13CLEF-QA4MRE.pdf

The (English only) QA4MRE challenge which was run as a Lab at CLEF 2011-2013.
The main objective of this exercise is to develop a methodology for evaluating
Machine Reading systems through Question Answering and Reading Comprehension
Tests. Systems should be able to extract knowledge from large volumes of text
and use this knowledge to answer questions. Four different tasks have been
organized during these years: Main Task, Processing Modality and Negation for
Machine Reading, Machine Reading of Biomedical Texts about Alzheimer's disease,
and Entrance Exam.

Homepage: http://nlp.uned.es/clef-qa/repository/qa4mre.php
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{Peas2013QA4MRE2O,
    title={QA4MRE 2011-2013: Overview of Question Answering for Machine Reading Evaluation},
    author={Anselmo Pe{\~n}as and Eduard H. Hovy and Pamela Forner and {\'A}lvaro Rodrigo and Richard F. E. Sutcliffe and Roser Morante},
    booktitle={CLEF},
    year={2013}
}
"""  # noqa: W605


class QA4MRE(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "qa4mre"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        # `qa4mre` only has train data so we use it for the test docs.
        return map(self._process_doc, self.dataset["train"])

    def _process_doc(self, doc):
        choices = doc["answer_options"]["answer_str"]
        out_doc = {
            "source": doc["document_str"].strip().replace("'", "'"),
            "query": doc["question_str"],
            "choices": choices,
            "gold": int(doc["correct_answer_id"]) - 1,
        }
        return out_doc

    def doc_to_text(self, doc):
        return "{}\nQuestion: {}\nAnswer:".format(doc["source"], doc["query"])

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["source"] + " " + doc["query"]


class QA4MRE_2011(QA4MRE):
    DATASET_NAME = "2011.main.EN"


class QA4MRE_2012(QA4MRE):
    DATASET_NAME = "2012.main.EN"


class QA4MRE_2013(QA4MRE):
    DATASET_NAME = "2013.main.EN"
