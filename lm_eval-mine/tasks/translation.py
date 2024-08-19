"""
NOTE: This file implements translation tasks using datasets from WMT conferences,
provided by sacrebleu. Traditionally they are evaluated with BLEU scores. TER
and CHRF are other options.

We defer citations and descriptions of the many translations tasks used
here to the SacreBLEU repo from which we've obtained the datasets:
https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/dataset.py

Homepage: https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/dataset.py
"""
import pycountry
from pprint import pprint
from sacrebleu import sacrebleu
from lm_eval import metrics
from lm_eval.base import Task, rf
from typing import List

try:
    import nagisa

    HAS_NAGISA = True
except ImportError:
    HAS_NAGISA = False

try:
    import jieba

    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False


_CITATION = """
@inproceedings{post-2018-call,
    title = "A Call for Clarity in Reporting {BLEU} Scores",
    author = "Post, Matt",
    booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
    month = oct,
    year = "2018",
    address = "Belgium, Brussels",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-6319",
    pages = "186--191",
}
"""


sacrebleu_datasets = sacrebleu.DATASETS


def create_tasks_from_benchmarks(benchmark_dict):
    """Creates a dictionary of tasks from a dict
    :param benchmark_dict: { dataset: [lang_pair, ...], }
    :return: {task_name: task}
        e.g. {wmt14-fr-en: Task, wmt16-de-en: Task}
    """

    def version_of(dataset, language_pair):
        if language_pair[-2:] in ["zh", "ja"]:
            return 1  # changed to use jieba/nagisa
        return 0

    return {
        f"{dataset}-{language_pair}": create_translation_task(
            dataset, language_pair, version_of(dataset, language_pair)
        )
        for dataset, language_pairs in benchmark_dict.items()
        for language_pair in language_pairs
    }


########################################
# Language Specifics
########################################


def zh_split(zh_text: List[str]) -> List[str]:
    """Chinese splitting"""
    if not HAS_JIEBA:
        raise ImportError(
            "Chinese text splitting requires the `jieba` package. "
            "Please install it with:\npip install jieba"
        )

    return [" ".join(jieba.cut(txt.strip())) for txt in zh_text]


def ja_split(ja_text: List[str]) -> List[str]:
    """Japanese splitting"""
    if not HAS_NAGISA:
        raise ImportError(
            "Japanese text splitting requires the `nagisa` package. "
            "Please install it with:\npip install nagisa"
        )

    return [" ".join(nagisa.tagging(txt.strip()).words) for txt in ja_text]


NO_SPACE_LANG = {"zh": zh_split, "ja": ja_split}

########################################
# Tasks
########################################


def create_translation_task(dataset, language_pair, version=0):
    class TranslationTask(GeneralTranslationTask):
        VERSION = version

        def __init__(self):
            super().__init__(dataset, language_pair)

    return TranslationTask


class GeneralTranslationTask(Task):
    VERSION = 0

    # e.g. ("wmt14", "fr-en")
    def __init__(self, sacrebleu_dataset, sacrebleu_language_pair=None):
        self.sacrebleu_dataset = sacrebleu_dataset
        self.sacrebleu_language_pair = sacrebleu_language_pair
        self.src_file = self.ref_file = self.src_data = self.ref_data = None

        super().__init__()

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        # This caches in the users home dir automatically
        self.src_file, self.ref_file = sacrebleu.download_test_set(
            self.sacrebleu_dataset, self.sacrebleu_language_pair
        )
        self.src_data, self.ref_data = [
            [line.rstrip() for line in sacrebleu.smart_open(file)]
            for file in (self.src_file, self.ref_file)
        ]

    def has_training_docs(self):
        """Whether the task has a training set"""
        # TODO In the future we could be more discerning. Some more recent tests have train and dev sets
        return False

    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return False

    def has_test_docs(self):
        """Whether the task has a test set"""
        return True

    def test_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return [
            {"src": src, "ref": ref} for src, ref in zip(self.src_data, self.ref_data)
        ]

    def doc_to_text(self, doc):
        language_codes = self.sacrebleu_language_pair.split("-")
        src_lang = code_to_language(language_codes[0])
        tar_lang = code_to_language(language_codes[1])
        return f"{src_lang} phrase: " + doc["src"] + f"\n{tar_lang} phrase:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["src"]

    def doc_to_target(self, doc):
        # This shows a single target, though there may be multiple targets in a lang test
        return " " + doc["ref"] if isinstance(doc["ref"], str) else doc["ref"][0]

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
        return rf.greedy_until(ctx, ["\n"])

    def process_results(self, doc, results):
        # Add spaces between words for BLEU score calculation of target languages like Chinese
        tar_lang_code = self.sacrebleu_language_pair.split("-")[-1]
        if tar_lang_code in NO_SPACE_LANG:
            doc["ref"] = NO_SPACE_LANG[tar_lang_code]([doc["ref"]])[0]
            results = NO_SPACE_LANG[tar_lang_code](results)

        # These metrics are corpus-level not sentence level, so we'll hide the
        # results in this dict and compute the corpus score in the aggregate method
        ref_pred = (doc["ref"], results)
        return {
            "bleu": ref_pred,
            "chrf": ref_pred,
            "ter": ref_pred,
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "bleu": metrics.bleu,
            "chrf": metrics.chrf,
            "ter": metrics.ter,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "bleu": True,
            "chrf": True,
            "ter": False,
        }

    def __str__(self):
        language_codes = self.sacrebleu_language_pair.split("-")
        src_lang = code_to_language(language_codes[0])
        tar_lang = code_to_language(language_codes[1])
        return f"{self.sacrebleu_dataset.upper()} {src_lang} to {tar_lang} Task"


########################################
# Util
########################################


def code_to_language(code):
    # key is alpha_2 or alpha_3 depending on the code length
    language_tuple = pycountry.languages.get(**{f"alpha_{len(code)}": code})
    return language_tuple.name
