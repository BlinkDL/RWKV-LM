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
"""MATH dataset."""


import json
import os
import pathlib

import datasets


_CITATION = """\
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the Math Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""

_DESCRIPTION = """\
MATH is a dataset of 12,500 challenging competition mathematics problems. Each
problem in Math has a full step-by-step solution which can be used to teach
models to generate answer derivations and explanations.
"""

_HOMEPAGE = "https://github.com/hendrycks/math"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URLS = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"

_NAMES = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


class HendrycksMath(datasets.GeneratorBasedBuilder):
    """MATH is a dataset of 12,500 challenging competition mathematics problems."""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=name, version=version, description=name)
        for name, version in zip(_NAMES, [VERSION] * len(_NAMES))
    ]

    def _info(self):
        features = datasets.Features(
            {
                "problem": datasets.Value("string"),
                "level": datasets.Value("string"),
                "type": datasets.Value("string"),
                "solution": datasets.Value("string"),
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
        urls = _URLS
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "basepath": os.path.join(
                        data_dir, "MATH", "train", self.config.name
                    ),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "basepath": os.path.join(
                        data_dir, "MATH", "test", self.config.name
                    ),
                    "split": "test",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, basepath, split):
        key = 0
        for file in sorted(pathlib.Path(basepath).iterdir()):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                yield key, {
                    "problem": data["problem"],
                    "level": data["level"],
                    "type": data["type"],
                    "solution": data["solution"],
                }
                key += 1
