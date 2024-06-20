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
"""Unscramble dataset."""


import json
import os

import datasets


_CITATION = """\
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

_DESCRIPTION = """\
Unscramble is a small battery of 5 “character manipulation” tasks. Each task
involves giving the model a word distorted by some combination of scrambling,
addition, or deletion of characters, and asking it to recover the original word.
"""

_HOMEPAGE = "https://github.com/openai/gpt-3/tree/master/data"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_BASE_URL = "https://raw.githubusercontent.com/openai/gpt-3/master/data"


_DESCRIPTIONS = {
    "mid_word_1_anagrams": "Anagrams of all but the first and last letter.",
    "mid_word_2_anagrams": "Anagrams of all but the first and last 2 letters.",
    "cycle_letters_in_word": "Cycle letters in the word.",
    "random_insertion_in_word": "Random insertions in the word that must be removed.",
    "reversed_words": "Words spelled backwards that must be reversed.",
}
_NAMES = _DESCRIPTIONS.keys()


class Unscramble(datasets.GeneratorBasedBuilder):
    """Unscramble is a small battery of 5 “character manipulation” tasks."""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=name, version=version, description=_DESCRIPTIONS[name]
        )
        for name, version in zip(_NAMES, [VERSION] * len(_NAMES))
    ]

    def _info(self):
        features = datasets.Features(
            {
                "context": datasets.Value("string"),
                "completion": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = os.path.join(_BASE_URL, f"{self.config.name}.jsonl.gz")
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                yield key, {
                    "context": data["context"],
                    "completion": data["completion"],
                }
