"""
Language Models are Few-Shot Learners
https://arxiv.org/pdf/2005.14165.pdf

A small battery of 10 tests that involve asking language models a simple arithmetic
problem in natural language.

Homepage: https://github.com/openai/gpt-3/tree/master/data
"""
import inspect
import lm_eval.datasets.arithmetic.arithmetic
from lm_eval.base import Task, rf
from lm_eval.metrics import mean


_CITATION = """
@inproceedings{NEURIPS2020_1457c0d6,
    author = {Brown, Tom and Mann, Benjamin and Ryder, Nick and Subbiah, Melanie and Kaplan, Jared D and Dhariwal, Prafulla and Neelakantan, Arvind and Shyam, Pranav and Sastry, Girish and Askell, Amanda and Agarwal, Sandhini and Herbert-Voss, Ariel and Krueger, Gretchen and Henighan, Tom and Child, Rewon and Ramesh, Aditya and Ziegler, Daniel and Wu, Jeffrey and Winter, Clemens and Hesse, Chris and Chen, Mark and Sigler, Eric and Litwin, Mateusz and Gray, Scott and Chess, Benjamin and Clark, Jack and Berner, Christopher and McCandlish, Sam and Radford, Alec and Sutskever, Ilya and Amodei, Dario},
    booktitle = {Advances in Neural Information Processing Systems},
    editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
    pages = {1877--1901},
    publisher = {Curran Associates, Inc.},
    title = {Language Models are Few-Shot Learners},
    url = {https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf},
    volume = {33},
    year = {2020}
}
"""


class Arithmetic(Task):
    VERSION = 0
    DATASET_PATH = inspect.getfile(lm_eval.datasets.arithmetic.arithmetic)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return NotImplemented

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return NotImplemented

    def doc_to_text(self, doc):
        return doc["context"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        return doc["completion"]

    def construct_requests(self, doc, ctx):
        ll, is_prediction = rf.loglikelihood(ctx, doc["completion"])
        return is_prediction

    def process_results(self, doc, results):
        (is_prediction,) = results
        return {"acc": is_prediction}

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {"acc": True}


class Arithmetic2DPlus(Arithmetic):
    DATASET_NAME = "arithmetic_2da"


class Arithmetic2DMinus(Arithmetic):
    DATASET_NAME = "arithmetic_2ds"


class Arithmetic3DPlus(Arithmetic):
    DATASET_NAME = "arithmetic_3da"


class Arithmetic3DMinus(Arithmetic):
    DATASET_NAME = "arithmetic_3ds"


class Arithmetic4DPlus(Arithmetic):
    DATASET_NAME = "arithmetic_4da"


class Arithmetic4DMinus(Arithmetic):
    DATASET_NAME = "arithmetic_4ds"


class Arithmetic5DPlus(Arithmetic):
    DATASET_NAME = "arithmetic_5da"


class Arithmetic5DMinus(Arithmetic):
    DATASET_NAME = "arithmetic_5ds"


class Arithmetic2DMultiplication(Arithmetic):
    DATASET_NAME = "arithmetic_2dm"


class Arithmetic1DComposite(Arithmetic):
    DATASET_NAME = "arithmetic_1dc"
