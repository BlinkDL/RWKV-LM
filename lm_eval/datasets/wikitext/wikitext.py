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
#
# NOTE: This is a modified version of https://github.com/huggingface/datasets/blob/master/datasets/wikitext/wikitext.py
# that returns Wiki pages instead of Wiki text line-by-line.
"""WikiText Dataset."""


import os

import datasets


_CITATION = """\
@misc{merity2016pointer,
      title={Pointer Sentinel Mixture Models},
      author={Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
      year={2016},
      eprint={1609.07843},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
 The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified
 Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike
 License.
"""
_HOMEPAGE = "https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/"
_LICENSE = "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)"
_DATA_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext"


class WikitextConfig(datasets.BuilderConfig):
    """BuilderConfig for GLUE."""

    def __init__(self, data_url, **kwargs):
        """BuilderConfig for Wikitext
        Args:
          data_url: `string`, url to the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikitextConfig, self).__init__(
            version=datasets.Version(
                "1.0.0",
            ),
            **kwargs,
        )
        self.data_url = data_url


class Wikitext(datasets.GeneratorBasedBuilder):
    """TODO(wikitext_103): Short description of my dataset."""

    # TODO(wikitext_103): Set up version.
    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        WikitextConfig(
            name="wikitext-103-v1",
            data_url=_DATA_URL + "/" + "wikitext-103-v1.zip",
            description="Word level dataset. No processing is needed other than replacing newlines with <eos> tokens.",
        ),
        WikitextConfig(
            name="wikitext-2-v1",
            data_url=_DATA_URL + "/" + "wikitext-2-v1.zip",
            description="Word level dataset. No processing is needed other than replacing newlines with <eos> tokens.",
        ),
        WikitextConfig(
            name="wikitext-103-raw-v1",
            data_url=_DATA_URL + "/" + "wikitext-103-raw-v1.zip",
            description="Raw level dataset: the raw tokens before the addition of <unk> tokens. "
            "They should only be used for character level work or for creating newly derived datasets.",
        ),
        WikitextConfig(
            name="wikitext-2-raw-v1",
            data_url=_DATA_URL + "/" + "wikitext-2-raw-v1.zip",
            description="Raw level dataset: the raw tokens before the addition of <unk> tokens. "
            "They should only be used for character level work or for creating newly derived datasets.",
        ),
    ]

    def _info(self):
        # TODO(wikitext): Specifies the datasets.DatasetInfo object
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "page": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(wikitext): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        if self.config.name == "wikitext-103-v1":
            data_file = dl_manager.download_and_extract(self.config.data_url)
            data_dir = os.path.join(data_file, "wikitext-103")
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, "wiki.test.tokens"),
                        "split": "test",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, "wiki.train.tokens"),
                        "split": "train",
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "data_file": os.path.join(data_dir, "wiki.valid.tokens"),
                        "split": "valid",
                    },
                ),
            ]
        else:
            if self.config.name == "wikitext-103-raw-v1":
                data_file = dl_manager.download_and_extract(self.config.data_url)
                data_dir = os.path.join(data_file, "wikitext-103-raw")
                return [
                    datasets.SplitGenerator(
                        name=datasets.Split.TEST,
                        gen_kwargs={
                            "data_file": os.path.join(data_dir, "wiki.test.raw"),
                            "split": "test",
                        },
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.TRAIN,
                        gen_kwargs={
                            "data_file": os.path.join(data_dir, "wiki.train.raw"),
                            "split": "train",
                        },
                    ),
                    datasets.SplitGenerator(
                        name=datasets.Split.VALIDATION,
                        gen_kwargs={
                            "data_file": os.path.join(data_dir, "wiki.valid.raw"),
                            "split": "valid",
                        },
                    ),
                ]
            else:
                if self.config.name == "wikitext-2-raw-v1":
                    data_file = dl_manager.download_and_extract(self.config.data_url)
                    data_dir = os.path.join(data_file, "wikitext-2-raw")
                    return [
                        datasets.SplitGenerator(
                            name=datasets.Split.TEST,
                            gen_kwargs={
                                "data_file": os.path.join(data_dir, "wiki.test.raw"),
                                "split": "test",
                            },
                        ),
                        datasets.SplitGenerator(
                            name=datasets.Split.TRAIN,
                            gen_kwargs={
                                "data_file": os.path.join(data_dir, "wiki.train.raw"),
                                "split": "train",
                            },
                        ),
                        datasets.SplitGenerator(
                            name=datasets.Split.VALIDATION,
                            gen_kwargs={
                                "data_file": os.path.join(data_dir, "wiki.valid.raw"),
                                "split": "valid",
                            },
                        ),
                    ]
                else:
                    if self.config.name == "wikitext-2-v1":
                        data_file = dl_manager.download_and_extract(
                            self.config.data_url
                        )
                        data_dir = os.path.join(data_file, "wikitext-2")
                        return [
                            datasets.SplitGenerator(
                                name=datasets.Split.TEST,
                                gen_kwargs={
                                    "data_file": os.path.join(
                                        data_dir, "wiki.test.tokens"
                                    ),
                                    "split": "test",
                                },
                            ),
                            datasets.SplitGenerator(
                                name=datasets.Split.TRAIN,
                                gen_kwargs={
                                    "data_file": os.path.join(
                                        data_dir, "wiki.train.tokens"
                                    ),
                                    "split": "train",
                                },
                            ),
                            datasets.SplitGenerator(
                                name=datasets.Split.VALIDATION,
                                gen_kwargs={
                                    "data_file": os.path.join(
                                        data_dir, "wiki.valid.tokens"
                                    ),
                                    "split": "valid",
                                },
                            ),
                        ]

    def _generate_examples(self, data_file, split):
        """Yields examples."""
        with open(data_file, encoding="utf-8") as f:
            key = 0
            ret = []
            data = f.read().split("\n")
            for line in data:
                rline = line.replace("= = =", "===").replace("= =", "==").strip()
                if rline.startswith("= ") and rline.strip().endswith(" ="):
                    page = "\n".join(ret)
                    if page.strip():
                        yield key, {"page": page}
                        key += 1
                    ret = []
                ret.append(line)
            page = "\n".join(ret)
            yield key, {"page": page}
