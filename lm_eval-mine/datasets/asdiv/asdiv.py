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
"""ASDIV dataset."""


import os
import xml.etree.ElementTree as ET

import datasets


_CITATION = """\
@misc{miao2021diverse,
    title={A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers},
    author={Shen-Yun Miao and Chao-Chun Liang and Keh-Yih Su},
    year={2021},
    eprint={2106.15772},
    archivePrefix={arXiv},
    primaryClass={cs.AI}
}
"""

_DESCRIPTION = """\
ASDiv (Academia Sinica Diverse MWP Dataset) is a diverse (in terms of both language
patterns and problem types) English math word problem (MWP) corpus for evaluating
the capability of various MWP solvers. Existing MWP corpora for studying AI progress
remain limited either in language usage patterns or in problem types. We thus present
a new English MWP corpus with 2,305 MWPs that cover more text patterns and most problem
types taught in elementary school. Each MWP is annotated with its problem type and grade
level (for indicating the level of difficulty).
"""

_HOMEPAGE = "https://github.com/chaochun/nlu-asdiv-dataset"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URLS = "https://github.com/chaochun/nlu-asdiv-dataset/archive/55790e5270bb91ccfa5053194b25732534696b50.zip"


class ASDiv(datasets.GeneratorBasedBuilder):
    """ASDiv: A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers"""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="asdiv",
            version=VERSION,
            description="A diverse corpus for evaluating and developing english math word problem solvers",
        )
    ]

    def _info(self):
        features = datasets.Features(
            {
                "body": datasets.Value("string"),
                "question": datasets.Value("string"),
                "solution_type": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "formula": datasets.Value("string"),
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
        base_filepath = "nlu-asdiv-dataset-55790e5270bb91ccfa5053194b25732534696b50"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, base_filepath, "dataset", "ASDiv.xml"
                    ),
                    "split": datasets.Split.VALIDATION,
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        tree = ET.parse(filepath)
        root = tree.getroot()
        for key, problem in enumerate(root.iter("Problem")):
            yield key, {
                "body": problem.find("Body").text,
                "question": problem.find("Question").text,
                "solution_type": problem.find("Solution-Type").text,
                "answer": problem.find("Answer").text,
                "formula": problem.find("Formula").text,
            }
