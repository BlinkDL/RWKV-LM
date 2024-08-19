"""
Semantic Parsing on Freebase from Question-Answer Pairs
https://cs.stanford.edu/~pliang/papers/freebase-emnlp2013.pdf

WebQuestions is a benchmark for question answering. The dataset consists of 6,642
question/answer pairs. The questions are supposed to be answerable by Freebase, a
large knowledge graph. The questions are mostly centered around a single named entity.
The questions are popular ones asked on the web (at least in 2013).

Homepage: https://worksheets.codalab.org/worksheets/0xba659fe363cb46e7a505c5b6a774dc8a
"""
from lm_eval.base import rf, Task
from lm_eval.metrics import mean


_CITATION = """
@inproceedings{berant-etal-2013-semantic,
    title = "Semantic Parsing on {F}reebase from Question-Answer Pairs",
    author = "Berant, Jonathan  and
      Chou, Andrew  and
      Frostig, Roy  and
      Liang, Percy",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D13-1160",
    pages = "1533--1544",
}
"""


class WebQs(Task):
    VERSION = 0
    DATASET_PATH = "web_questions"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "Question: " + doc["question"] + "\nAnswer:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["question"]

    def doc_to_target(self, doc):
        # this picks one answer to be the "correct" one, despite sometimes
        # multiple correct answers being possible.
        # TODO: make sure we're actually handling multi-answer correctly
        return " " + doc["answers"][0]

    def _remove_prefixes(self, aliases):
        # Optimization: Remove any alias that has a strict prefix elsewhere in the list
        # we can do this because if the prefix is acceptable by isgreedy, we can stop looking
        aliases.sort()
        ret = [aliases[0]]
        for alias in aliases[1:]:
            if not alias.startswith(ret[-1]):
                ret.append(alias)

        return ret

    def construct_requests(self, doc, ctx):
        ret = []
        for alias in self._remove_prefixes(doc["answers"]):
            _, is_prediction = rf.loglikelihood(ctx, " " + alias)
            ret.append(is_prediction)
        return ret

    def process_results(self, doc, results):
        return {"acc": float(any(results))}

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {"acc": True}
