"""
ASDiv: A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers
https://arxiv.org/abs/2106.15772

ASDiv (Academia Sinica Diverse MWP Dataset) is a diverse (in terms of both language
patterns and problem types) English math word problem (MWP) corpus for evaluating
the capability of various MWP solvers. Existing MWP corpora for studying AI progress
remain limited either in language usage patterns or in problem types. We thus present
a new English MWP corpus with 2,305 MWPs that cover more text patterns and most problem
types taught in elementary school. Each MWP is annotated with its problem type and grade
level (for indicating the level of difficulty).

NOTE: We currently ignore formulas for answer generation.

Homepage: https://github.com/chaochun/nlu-asdiv-dataset
"""
import inspect
import lm_eval.datasets.asdiv.asdiv
from lm_eval.base import rf, Task
from lm_eval.metrics import mean


_CITATION = """
@misc{miao2021diverse,
    title={A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers},
    author={Shen-Yun Miao and Chao-Chun Liang and Keh-Yih Su},
    year={2021},
    eprint={2106.15772},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
"""


class Asdiv(Task):
    VERSION = 0
    DATASET_PATH = inspect.getfile(lm_eval.datasets.asdiv.asdiv)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        raise NotImplementedError("This dataset has no training docs")

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        raise NotImplementedError("This dataset has no test docs")

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        assert num_fewshot == 0, "ASDiv is intended only for the zero-shot setting."
        return super().fewshot_context(
            doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
        )

    def doc_to_text(self, doc):
        # TODO: add solution-type
        return doc["body"] + "\n" + "Question:" + doc["question"] + "\n" + "Answer:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["body"] + " " + doc["question"]

    def doc_to_target(self, doc):
        # TODO: add formula

        answer = doc["answer"].split(" (")[0]
        return " " + answer

    def construct_requests(self, doc, ctx):
        ll, is_greedy = rf.loglikelihood(ctx, self.doc_to_target(doc))
        return ll, is_greedy

    def process_results(self, doc, results):
        ll, is_greedy = results

        return {"acc": int(is_greedy)}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
