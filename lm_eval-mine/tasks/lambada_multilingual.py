"""
The LAMBADA (OpenAI) dataset: Word prediction requiring a broad discourse context∗
https://arxiv.org/pdf/1606.06031.pdf

The LAMBADA OpenAI dataset machine-translated to other languages.
LAMBADA is a dataset to evaluate the capabilities of computational models for text
understanding by means of a word prediction task. LAMBADA is a collection of narrative
passages sharing the characteristic that human subjects are able to guess their last
word if they are exposed to the whole passage, but not if they only see the last
sentence preceding the target word. To succeed on LAMBADA, computational models
cannot simply rely on local context, but must be able to keep track of information
in the broader discourse.

Homepage: https://zenodo.org/record/2630551#.X4Xzn5NKjUI

Reference (OpenAI): https://github.com/openai/gpt-2/issues/131#issuecomment-497136199
"""
from .lambada import LambadaOpenAI


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


class LambadaOpenAIMultilingualEnglish(LambadaOpenAI):
    VERSION = 0
    DATASET_NAME = "en"


class LambadaOpenAIMultilingualFrench(LambadaOpenAI):
    VERSION = 0
    DATASET_NAME = "fr"


class LambadaOpenAIMultilingualGerman(LambadaOpenAI):
    VERSION = 0
    DATASET_NAME = "de"


class LambadaOpenAIMultilingualItalian(LambadaOpenAI):
    VERSION = 0
    DATASET_NAME = "it"


class LambadaOpenAIMultilingualSpanish(LambadaOpenAI):
    VERSION = 0
    DATASET_NAME = "es"


LANG_CLASSES = [
    LambadaOpenAIMultilingualEnglish,
    LambadaOpenAIMultilingualFrench,
    LambadaOpenAIMultilingualGerman,
    LambadaOpenAIMultilingualItalian,
    LambadaOpenAIMultilingualSpanish,
]


def construct_tasks():
    tasks = {}
    for lang_class in LANG_CLASSES:
        tasks[f"lambada_openai_mt_{lang_class.DATASET_NAME}"] = lang_class
    return tasks
