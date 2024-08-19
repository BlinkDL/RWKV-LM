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
"""MuTual dataset."""


import json
import os
from pathlib import Path

import datasets


_CITATION = """\
@inproceedings{mutual,
    title = "MuTual: A Dataset for Multi-Turn Dialogue Reasoning",
    author = "Cui, Leyang  and Wu, Yu and Liu, Shujie and Zhang, Yue and Zhou, Ming" ,
    booktitle = "Proceedings of the 58th Conference of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
"""

_DESCRIPTION = """\
MuTual is a retrieval-based dataset for multi-turn dialogue reasoning, which is
modified from Chinese high school English listening comprehension test data.
"""

_HOMEPAGE = "https://github.com/Nealcly/MuTual"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URLS = "https://github.com/Nealcly/MuTual/archive/master.zip"


class Mutual(datasets.GeneratorBasedBuilder):
    """MuTual: A Dataset for Multi-Turn Dialogue Reasoning"""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="mutual", version=VERSION, description="The MuTual dataset."
        ),
        datasets.BuilderConfig(
            name="mutual_plus",
            version=VERSION,
            description="MuTualPlus is a more difficult MuTual that replaces positive responses with a safe responses.",
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "answers": datasets.Value("string"),
                "options": datasets.features.Sequence(datasets.Value("string")),
                "article": datasets.Value("string"),
                "id": datasets.Value("string"),
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
        urls = _URLS
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "basepath": os.path.join(
                        data_dir, "MuTual-master", "data", self.config.name, "train"
                    ),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "basepath": os.path.join(
                        data_dir, "MuTual-master", "data", self.config.name, "test"
                    ),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "basepath": os.path.join(
                        data_dir, "MuTual-master", "data", self.config.name, "dev"
                    ),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, basepath, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        key = 0
        for file in sorted(Path(basepath).iterdir()):
            if file.suffix != ".txt":
                continue
            with open(file, "r", encoding="utf-8") as f:
                data_str = f.read()
                # Ignore the occasional empty file.
                if not data_str:
                    continue
                data = json.loads(data_str)
                yield key, {
                    "answers": data["answers"],
                    "options": data["options"],
                    "article": data["article"],
                    "id": data["id"],
                }
                key += 1
