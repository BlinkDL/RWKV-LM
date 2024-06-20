"""
The LAMBADA dataset: Word prediction requiring a broad discourse context∗
https://arxiv.org/pdf/1606.06031.pdf

LAMBADA is a dataset to evaluate the capabilities of computational models for text
understanding by means of a word prediction task. LAMBADA is a collection of narrative
passages sharing the characteristic that human subjects are able to guess their last
word if they are exposed to the whole passage, but not if they only see the last
sentence preceding the target word. To succeed on LAMBADA, computational models
cannot simply rely on local context, but must be able to keep track of information
in the broader discourse.

Homepage: https://zenodo.org/record/2630551#.X4Xzn5NKjUI
"""
import inspect
import lm_eval.datasets.lambada_openai.lambada_openai
from lm_eval.base import Task, rf
from lm_eval.metrics import mean, perplexity


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


class LambadaBase(Task):
    VERSION = None

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return doc["text"].rsplit(" ", 1)[0]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    def doc_to_target(self, doc):
        return " " + doc["text"].rsplit(" ", 1)[1]

    def construct_requests(self, doc, ctx):
        ll, is_greedy = rf.loglikelihood(ctx, self.doc_to_target(doc))

        return ll, is_greedy

    def process_results(self, doc, results):
        ll, is_greedy = results

        return {"ppl": ll, "acc": int(is_greedy)}

    def aggregation(self):
        return {"ppl": perplexity, "acc": mean}

    def higher_is_better(self):
        return {"ppl": False, "acc": True}


class LambadaStandard(LambadaBase):
    """The LAMBADA task using the standard original LAMBADA dataset."""

    VERSION = 0
    DATASET_PATH = "lambada"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True


class LambadaOpenAI(LambadaBase):
    """The LAMBADA task using the LAMBADA OpenAI dataset, a modified version of the
    original LAMBADA dataset created by OpenAI for evaluating their GPT-2 model.

    Reference: https://github.com/openai/gpt-2/issues/131#issuecomment-497136199
    """

    VERSION = 0
    DATASET_PATH = inspect.getfile(lm_eval.datasets.lambada_openai.lambada_openai)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False
