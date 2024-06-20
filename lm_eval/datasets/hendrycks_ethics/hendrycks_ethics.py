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
"""ETHICS dataset."""
# TODO: Add the `hard` dataset splits.


import csv
import os

import datasets


_CITATION = """\
@article{hendrycks2021ethics
    title={Aligning AI With Shared Human Values},
    author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021}
}
"""

_DESCRIPTION = """\
The ETHICS dataset is a benchmark that spans concepts in justice, well-being,
duties, virtues, and commonsense morality. Models predict widespread moral
judgments about diverse text scenarios. This requires connecting physical and
social world knowledge to value judgements, a capability that may enable us
to steer chatbot outputs or eventually regularize open-ended reinforcement
learning agents.
"""

_HOMEPAGE = "https://github.com/hendrycks/ethics"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URLS = "https://people.eecs.berkeley.edu/~hendrycks/ethics.tar"


class EthicsConfig(datasets.BuilderConfig):
    """BuilderConfig for Hendrycks ETHICS."""

    def __init__(self, prefix, features, **kwargs):
        """BuilderConfig for Hendrycks ETHICS.

        Args:
        prefix: *string*, prefix to add to the dataset name for path location.
        features: *list[string]*, list of the features that will appear in the
            feature dict.
        """
        # Version history:
        super().__init__(version=datasets.Version("0.0.1"), **kwargs)
        self.prefix = prefix
        self.features = features


class HendrycksEthics(datasets.GeneratorBasedBuilder):
    """The ETHICS dataset is a benchmark that spans concepts in justice, well-being, duties, virtues, and commonsense morality."""

    BUILDER_CONFIGS = [
        EthicsConfig(
            name="commonsense",
            prefix="cm",
            features=datasets.Features(
                {
                    "label": datasets.Value("int32"),
                    "input": datasets.Value("string"),
                    "is_short": datasets.Value("bool"),
                    "edited": datasets.Value("bool"),
                }
            ),
            description="The Commonsense subset contains examples focusing on moral standards and principles that most people intuitively accept.",
        ),
        EthicsConfig(
            name="deontology",
            prefix="deontology",
            features=datasets.Features(
                {
                    "group_id": datasets.Value("int32"),
                    "label": datasets.Value("int32"),
                    "scenario": datasets.Value("string"),
                    "excuse": datasets.Value("string"),
                }
            ),
            description="The Deontology subset contains examples focusing on whether an act is required, permitted, or forbidden according to a set of rules or constraints",
        ),
        EthicsConfig(
            name="justice",
            prefix="justice",
            features=datasets.Features(
                {
                    "group_id": datasets.Value("int32"),
                    "label": datasets.Value("int32"),
                    "scenario": datasets.Value("string"),
                }
            ),
            description="The Justice subset contains examples focusing on how a character treats another person",
        ),
        EthicsConfig(
            name="utilitarianism",
            prefix="util",
            features=datasets.Features(
                {
                    "activity": datasets.Value("string"),
                    "baseline": datasets.Value("string"),
                    "rating": datasets.Value("string"),  # Empty rating.
                }
            ),
            description="The Utilitarianism subset contains scenarios that should be ranked from most pleasant to least pleasant for the person in the scenario",
        ),
        EthicsConfig(
            name="virtue",
            prefix="virtue",
            features=datasets.Features(
                {
                    "group_id": datasets.Value("int32"),
                    "label": datasets.Value("int32"),
                    "scenario": datasets.Value("string"),
                    "trait": datasets.Value("string"),
                }
            ),
            description="The Virtue subset contains scenarios focusing on whether virtues or vices are being exemplified",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=f"{_DESCRIPTION}\n{self.config.description}",
            features=self.config.features,
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
                    "filepath": os.path.join(
                        data_dir,
                        "ethics",
                        self.config.name,
                        f"{self.config.prefix}_train.csv",
                    ),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "ethics",
                        self.config.name,
                        f"{self.config.prefix}_test.csv",
                    ),
                    "split": "test",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, newline="") as f:
            if self.config.name == "utilitarianism":
                contents = csv.DictReader(f, fieldnames=["activity", "baseline"])
            else:
                contents = csv.DictReader(f)
            # For subsets with grouped scenarios, tag them with an id.
            group_id = 0
            for key, row in enumerate(contents):
                if self.config.name == "deontology":
                    # Scenarios come in groups of 4.
                    if key % 4 == 0 and key != 0:
                        group_id += 1
                    yield key, {
                        "group_id": group_id,
                        "label": row["label"],
                        "scenario": row["scenario"],
                        "excuse": row["excuse"],
                    }
                elif self.config.name == "justice":
                    # Scenarios come in groups of 4.
                    if key % 4 == 0 and key != 0:
                        group_id += 1
                    yield key, {
                        "group_id": group_id,
                        "label": row["label"],
                        "scenario": row["scenario"],
                    }
                elif self.config.name == "commonsense":
                    yield key, {
                        "label": row["label"],
                        "input": row["input"],
                        "is_short": row["is_short"],
                        "edited": row["edited"],
                    }
                elif self.config.name == "virtue":
                    # Scenarios come in groups of 5.
                    if key % 5 == 0 and key != 0:
                        group_id += 1
                    scenario, trait = row["scenario"].split(" [SEP] ")
                    yield key, {
                        "group_id": group_id,
                        "label": row["label"],
                        "scenario": scenario,
                        "trait": trait,
                    }
                elif self.config.name == "utilitarianism":
                    yield key, {
                        "activity": row["activity"],
                        "baseline": row["baseline"],
                        "rating": "",
                    }
