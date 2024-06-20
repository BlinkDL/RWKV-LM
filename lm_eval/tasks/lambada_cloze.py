"""
The LAMBADA dataset: Word prediction requiring a broad discourse context∗
https://arxiv.org/pdf/1606.06031.pdf

Cloze-style LAMBADA dataset.
LAMBADA is a dataset to evaluate the capabilities of computational models for text
understanding by means of a word prediction task. LAMBADA is a collection of narrative
passages sharing the characteristic that human subjects are able to guess their last
word if they are exposed to the whole passage, but not if they only see the last
sentence preceding the target word. To succeed on LAMBADA, computational models
cannot simply rely on local context, but must be able to keep track of information
in the broader discourse.

Homepage: https://zenodo.org/record/2630551#.X4Xzn5NKjUI
"""
from lm_eval.tasks.lambada import LambadaOpenAI, LambadaStandard


_CITATION = """
@misc{
    author={Paperno, Denis and Kruszewski, Germán and Lazaridou, Angeliki and Pham, Quan Ngoc and Bernardi, Raffaella and Pezzelle, Sandro and Baroni, Marco and Boleda, Gemma and Fernández, Raquel},
    title={The LAMBADA dataset},
    DOI={10.5281/zenodo.2630551},
    publisher={Zenodo},
    year={2016},
    month={Aug}
}
"""


class LambadaStandardCloze(LambadaStandard):
    """Cloze-style LambadaStandard."""

    VERSION = 0

    def doc_to_text(self, doc):
        return doc["text"].rsplit(" ", 1)[0] + " ____. ->"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        return " " + doc["text"].rsplit(" ", 1)[1]


class LambadaOpenAICloze(LambadaOpenAI):
    """Cloze-style LambadaOpenAI."""

    VERSION = 0

    def doc_to_text(self, doc):
        return doc["text"].rsplit(" ", 1)[0] + " ____. ->"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        return " " + doc["text"].rsplit(" ", 1)[1]
