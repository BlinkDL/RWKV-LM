"""
Language Models are Few-Shot Learners
https://arxiv.org/pdf/2005.14165.pdf

Unscramble is a small battery of 5 “character manipulation” tasks. Each task
involves giving the model a word distorted by some combination of scrambling,
addition, or deletion of characters, and asking it to recover the original word.

Homepage: https://github.com/openai/gpt-3/tree/master/data
"""
import inspect
import lm_eval.datasets.unscramble.unscramble
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


class WordUnscrambleTask(Task):
    VERSION = 0
    DATASET_PATH = inspect.getfile(lm_eval.datasets.unscramble.unscramble)
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return doc["context"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["context"]

    def doc_to_target(self, doc):
        return doc["completion"]

    def construct_requests(self, doc, ctx):
        completion = rf.greedy_until(ctx, ["\n"])
        return completion

    def process_results(self, doc, results):
        pred = results[0]
        gold = doc["completion"]
        return {"acc": int(pred == gold)}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}


class Anagrams1(WordUnscrambleTask):
    DATASET_NAME = "mid_word_1_anagrams"


class Anagrams2(WordUnscrambleTask):
    DATASET_NAME = "mid_word_2_anagrams"


class CycleLetters(WordUnscrambleTask):
    DATASET_NAME = "cycle_letters_in_word"


class RandomInsertion(WordUnscrambleTask):
    DATASET_NAME = "random_insertion_in_word"


class ReversedWords(WordUnscrambleTask):
    DATASET_NAME = "reversed_words"
