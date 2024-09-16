"""
Pointer Sentinel Mixture Models
https://arxiv.org/pdf/1609.07843.pdf

The WikiText language modeling dataset is a collection of over 100 million tokens
extracted from the set of verified Good and Featured articles on Wikipedia.

NOTE: This `Task` is based on WikiText-2.

Homepage: https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
"""
import re
import inspect
import lm_eval.datasets.wikitext.wikitext
from lm_eval.base import PerplexityTask


_CITATION = """
@misc{merity2016pointer,
    title={Pointer Sentinel Mixture Models},
    author={Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
    year={2016},
    eprint={1609.07843},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string


class WikiText(PerplexityTask):
    VERSION = 1
    DATASET_PATH = inspect.getfile(lm_eval.datasets.wikitext.wikitext)
    DATASET_NAME = "wikitext-2-raw-v1"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        return doc["page"]

    def doc_to_target(self, doc):
        return wikitext_detokenizer(doc)

    def should_decontaminate(self):
        return True

    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(re.split(r"\s+", doc))
