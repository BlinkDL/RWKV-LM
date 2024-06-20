# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pile dataset."""


import json

import datasets


_CITATION = """\
@article{pile,
  title={The {P}ile: An 800GB Dataset of Diverse Text for Language Modeling},
  author={Gao, Leo and Biderman, Stella and Black, Sid and Golding, Laurence and Hoppe, Travis and Foster, Charles and Phang, Jason and He, Horace and Thite, Anish and Nabeshima, Noa and Presser, Shawn and Leahy, Connor},
  journal={arXiv preprint arXiv:2101.00027},
  year={2020}
}
"""

_DESCRIPTION = """\
The Pile is a 825 GiB diverse, open source language modeling data set that consists
of 22 smaller, high-quality datasets combined together. To score well on Pile
BPB (bits per byte), a model must be able to understand many disparate domains
including books, github repositories, webpages, chat logs, and medical, physics,
math, computer science, and philosophy papers.
"""

_HOMEPAGE = "https://pile.eleuther.ai/"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URLS = {
    "validation": "http://eaidata.bmk.sh/data/pile/val.jsonl.zst",
    "test": "http://eaidata.bmk.sh/data/pile/test.jsonl.zst",
}

_NAMES = {
    "pile_arxiv": "ArXiv",
    "pile_books3": "Books3",
    "pile_bookcorpus2": "BookCorpus2",
    "pile_dm-mathematics": "DM Mathematics",
    "pile_enron": "Enron Emails",
    "pile_europarl": "EuroParl",
    "pile_freelaw": "FreeLaw",
    "pile_github": "Github",
    "pile_gutenberg": "Gutenberg (PG-19)",
    "pile_hackernews": "HackerNews",
    "pile_nih-exporter": "NIH ExPorter",
    "pile_opensubtitles": "OpenSubtitles",
    "pile_openwebtext2": "OpenWebText2",
    "pile_philpapers": "PhilPapers",
    "pile_pile-cc": "Pile-CC",
    "pile_pubmed-abstracts": "PubMed Abstracts",
    "pile_pubmed-central": "PubMed Central",
    "pile_stackexchange": "StackExchange",
    "pile_upsto": "USPTO Backgrounds",
    "pile_ubuntu-irc": "Ubuntu IRC",
    "pile_wikipedia": "Wikipedia (en)",
    "pile_youtubesubtitles": "YoutubeSubtitles",
}


class Pile(datasets.GeneratorBasedBuilder):
    """The Pile is a 825 GiB diverse, open source language modeling dataset."""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=name, version=version, description=_NAMES[name])
        for name, version in zip(_NAMES.keys(), [VERSION] * len(_NAMES))
    ]

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=f"{_DESCRIPTION}\n{self.config.description}",
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = {"validation": _URLS["validation"], "test": _URLS["test"]}
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": data_dir["test"], "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["validation"],
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                if data["meta"]["pile_set_name"] == _NAMES[self.config.name]:
                    yield key, {
                        "text": data["text"],
                    }
