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
# TODO: Address all TODOs and remove all explanatory comments
"""QuAC dataset."""


import json

import datasets


_CITATION = """\
@article{choi2018quac,
    title={Quac: Question answering in context},
    author={Choi, Eunsol and He, He and Iyyer, Mohit and Yatskar, Mark and Yih, Wen-tau and Choi, Yejin and Liang, Percy and Zettlemoyer, Luke},
    journal={arXiv preprint arXiv:1808.07036},
    year={2018}
}
"""

_DESCRIPTION = """\
Question Answering in Context (QuAC) is a dataset for modeling, understanding, and
participating in information seeking dialog. Data instances consist of an interactive
dialog between two crowd workers: (1) a student who poses a sequence of freeform
questions to learn as much as possible about a hidden Wikipedia text, and (2)
a teacher who answers the questions by providing short excerpts (spans) from the text.
"""

_HOMEPAGE = "https://quac.ai/"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URLS = {
    "train": "https://s3.amazonaws.com/my89public/quac/train_v0.2.json",
    "validation": "https://s3.amazonaws.com/my89public/quac/val_v0.2.json",
}


class Quac(datasets.GeneratorBasedBuilder):
    """Question Answering in Context (QuAC) is a dataset for modeling, understanding, and  participating in information seeking dialog."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="quac", version=VERSION, description="The QuAC dataset"
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "title": datasets.Value("string"),
                "section_title": datasets.Value("string"),
                "paragraph": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
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
        urls = {"train": _URLS["train"], "validation": _URLS["validation"]}
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": data_dir["validation"], "split": "validation"},
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)["data"]
            key = 0
            for row in data:
                paragraph = row["paragraphs"][0]["context"].replace("CANNOTANSWER", "")
                qas = row["paragraphs"][0]["qas"]
                qa_pairs = [(qa["question"], qa["answers"][0]["text"]) for qa in qas]
                for (question, answer) in qa_pairs:
                    # Yields examples as (key, example) tuples
                    yield key, {
                        "title": row["title"],
                        "section_title": row["section_title"],
                        "paragraph": paragraph,
                        "question": question,
                        "answer": answer,
                    }
                    key += 1
