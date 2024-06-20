"""
The Pile: An 800GB Dataset of Diverse Text for Language Modeling
https://arxiv.org/pdf/2101.00027.pdf

The Pile is a 825 GiB diverse, open source language modelling data set that consists
of 22 smaller, high-quality datasets combined together. To score well on Pile
BPB (bits per byte), a model must be able to understand many disparate domains
including books, github repositories, webpages, chat logs, and medical, physics,
math, computer science, and philosophy papers.

Homepage: https://pile.eleuther.ai/
"""
import inspect
import lm_eval.datasets.pile.pile
from lm_eval.base import PerplexityTask


_CITATION = """
@article{pile,
  title={The {P}ile: An 800GB Dataset of Diverse Text for Language Modeling},
  author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020}
}
"""


class PilePerplexityTask(PerplexityTask):
    VERSION = 1
    DATASET_PATH = inspect.getfile(lm_eval.datasets.pile.pile)
    DATASET_NAME = None

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        for doc in self.dataset["validation"]:
            yield doc["text"]

    def test_docs(self):
        for doc in self.dataset["test"]:
            yield doc["text"]


class PileArxiv(PilePerplexityTask):
    DATASET_NAME = "pile_arxiv"


class PileBooks3(PilePerplexityTask):
    DATASET_NAME = "pile_books3"


class PileBookCorpus2(PilePerplexityTask):
    DATASET_NAME = "pile_bookcorpus2"


class PileDmMathematics(PilePerplexityTask):
    DATASET_NAME = "pile_dm-mathematics"


class PileEnron(PilePerplexityTask):
    DATASET_NAME = "pile_enron"


class PileEuroparl(PilePerplexityTask):
    DATASET_NAME = "pile_europarl"


class PileFreeLaw(PilePerplexityTask):
    DATASET_NAME = "pile_freelaw"


class PileGithub(PilePerplexityTask):
    DATASET_NAME = "pile_github"


class PileGutenberg(PilePerplexityTask):
    DATASET_NAME = "pile_gutenberg"


class PileHackernews(PilePerplexityTask):
    DATASET_NAME = "pile_hackernews"


class PileNIHExporter(PilePerplexityTask):
    DATASET_NAME = "pile_nih-exporter"


class PileOpenSubtitles(PilePerplexityTask):
    DATASET_NAME = "pile_opensubtitles"


class PileOpenWebText2(PilePerplexityTask):
    DATASET_NAME = "pile_openwebtext2"


class PilePhilPapers(PilePerplexityTask):
    DATASET_NAME = "pile_philpapers"


class PilePileCc(PilePerplexityTask):
    DATASET_NAME = "pile_pile-cc"


class PilePubmedAbstracts(PilePerplexityTask):
    DATASET_NAME = "pile_pubmed-abstracts"


class PilePubmedCentral(PilePerplexityTask):
    DATASET_NAME = "pile_pubmed-central"


class PileStackExchange(PilePerplexityTask):
    DATASET_NAME = "pile_stackexchange"


class PileUspto(PilePerplexityTask):
    DATASET_NAME = "pile_upsto"


class PileUbuntuIrc(PilePerplexityTask):
    DATASET_NAME = "pile_ubuntu-irc"


class PileWikipedia(PilePerplexityTask):
    DATASET_NAME = "pile_wikipedia"


class PileYoutubeSubtitles(PilePerplexityTask):
    DATASET_NAME = "pile_youtubesubtitles"
