"""
The Winograd Schema Challenge
http://commonsensereasoning.org/2011/papers/Levesque.pdf

A Winograd schema is a pair of sentences that differ in only one or two words
and that contain an ambiguity that is resolved in opposite ways in the two
sentences and requires the use of world knowledge and reasoning for its resolution.
The Winograd Schema Challenge 273 is a collection of 273 such Winograd schemas.

NOTE: This evaluation of Winograd Schema Challenge is based on `partial evaluation`
as described by Trinh & Le in Simple Method for Commonsense Reasoning (2018).
See: https://arxiv.org/abs/1806.0

Homepage: https://cs.nyu.edu/~davise/papers/WinogradSchemas/WS.html
"""
import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean


_CITATION = """
@inproceedings{ea01b9c0db064caca6986b925d75f2bb,
    title = "The winograd schema challenge",
    abstract = "In this paper, we present an alternative to the Turing Test that has some conceptual and practical advantages. A Wino-grad schema is a pair of sentences that differ only in one or two words and that contain a referential ambiguity that is resolved in opposite directions in the two sentences. We have compiled a collection of Winograd schemas, designed so that the correct answer is obvious to the human reader, but cannot easily be found using selectional restrictions or statistical techniques over text corpora. A contestant in the Winograd Schema Challenge is presented with a collection of one sentence from each pair, and required to achieve human-level accuracy in choosing the correct disambiguation.",
    author = "Levesque, {Hector J.} and Ernest Davis and Leora Morgenstern",
    year = "2012",
    language = "English (US)",
    isbn = "9781577355601",
    series = "Proceedings of the International Conference on Knowledge Representation and Reasoning",
    publisher = "Institute of Electrical and Electronics Engineers Inc.",
    pages = "552--561",
    booktitle = "13th International Conference on the Principles of Knowledge Representation and Reasoning, KR 2012",
    note = "13th International Conference on the Principles of Knowledge Representation and Reasoning, KR 2012 ; Conference date: 10-06-2012 Through 14-06-2012",
}
"""


class WinogradSchemaChallenge273(Task):
    VERSION = 0
    DATASET_PATH = "winograd_wsc"
    DATASET_NAME = "wsc273"

    upper_pronouns = [
        "A",
        "An",
        "The",
        "She",
        "He",
        "It",
        "They",
        "My",
        "His",
        "Her",
        "Their",
    ]

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        # The HF implementation of `wsc273` is not `partial evaluation` friendly.
        doc["text"] = doc["text"].replace("  ", " ")
        doc["options"][0] = self.__normalize_option(doc, doc["options"][0])
        doc["options"][1] = self.__normalize_option(doc, doc["options"][1])
        return doc

    def __normalize_option(self, doc, option):
        # Append `'s` to possessive determiner based options.
        if doc["pronoun"].lower() in ["my", "his", "her", "our", "their"]:
            option += "'s"
        # Appropriately lowercase the pronoun in the option.
        pronoun = option.split()[0]
        start_of_sentence = doc["text"][doc["pronoun_loc"] - 2] == "."
        if not start_of_sentence and pronoun in self.upper_pronouns:
            return option.replace(pronoun, pronoun.lower())
        return option

    def fewshot_examples(self, k, rnd):
        # NOTE: `super().fewshot_examples` samples from training docs which are
        # not available for this test-set-only dataset.

        if self._fewshot_docs is None:
            self._fewshot_docs = list(self.test_docs())

        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return self.partial_context(doc, doc["options"][doc["label"]])

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["text"]

    @classmethod
    def partial_context(cls, doc, option):
        # Substitute the pronoun in the original text with the specified
        # option and ignore everything after.
        return doc["text"][: doc["pronoun_loc"]] + option

    def doc_to_target(self, doc):
        return self.partial_target(doc)

    @classmethod
    def partial_target(cls, doc):
        # The target is everything after the document specified pronoun.
        start_index = doc["pronoun_loc"] + len(doc["pronoun"])
        return " " + doc["text"][start_index:].strip()

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        target = self.partial_target(doc)
        lls = []
        for option in doc["options"]:
            partial_ctx = self.partial_context(doc, option)
            full_ctx = self.append_context(ctx, partial_ctx)
            lls.append(rf.loglikelihood(full_ctx, target)[0])
        return lls

    @classmethod
    def append_context(cls, ctx, partial_ctx):
        ctx = ctx.split("\n\n")  # Each fewshot context is on its own new line.
        ctx.pop()  # Remove the correct context put in by `doc_to_text`.
        return "\n\n".join([*ctx, partial_ctx]) if ctx else partial_ctx

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        return {"acc": np.argmax(results) == doc["label"]}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"acc": mean}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {"acc": True}
