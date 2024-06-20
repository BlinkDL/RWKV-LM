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
"""CoQA dataset.

This `CoQA` adds the "additional_answers" feature that's missing in the original
datasets version:
https://github.com/huggingface/datasets/blob/master/datasets/coqa/coqa.py
"""


import json

import datasets


_CITATION = """\
@misc{reddy2018coqa,
    title={CoQA: A Conversational Question Answering Challenge},
    author={Siva Reddy and Danqi Chen and Christopher D. Manning},
    year={2018},
    eprint={1808.07042},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
CoQA is a large-scale dataset for building Conversational Question Answering
systems. The goal of the CoQA challenge is to measure the ability of machines to
understand a text passage and answer a series of interconnected questions that
appear in a conversation.
"""

_HOMEPAGE = "https://stanfordnlp.github.io/coqa/"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URLS = {
    "train": "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json",
    "validation": "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json",
}

# `additional_answers` are not available in the train set so we fill them with
# empty dicts of the same form.
_EMPTY_ADDITIONAL_ANSWER = {
    "0": [
        {
            "span_start": -1,
            "span_end": -1,
            "span_text": "",
            "input_text": "",
            "turn_id": -1,
        }
    ],
    "1": [
        {
            "span_start": -1,
            "span_end": -1,
            "span_text": "",
            "input_text": "",
            "turn_id": -1,
        }
    ],
    "2": [
        {
            "span_start": -1,
            "span_end": -1,
            "span_text": "",
            "input_text": "",
            "turn_id": -1,
        }
    ],
}


class Coqa(datasets.GeneratorBasedBuilder):
    """CoQA is a large-scale dataset for building Conversational Question Answering systems."""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="coqa", version=VERSION, description="The CoQA dataset."
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "source": datasets.Value("string"),
                "story": datasets.Value("string"),
                "questions": datasets.features.Sequence(
                    {
                        "input_text": datasets.Value("string"),
                        "turn_id": datasets.Value("int32"),
                    }
                ),
                "answers": datasets.features.Sequence(
                    {
                        "span_start": datasets.Value("int32"),
                        "span_end": datasets.Value("int32"),
                        "span_text": datasets.Value("string"),
                        "input_text": datasets.Value("string"),
                        "turn_id": datasets.Value("int32"),
                    }
                ),
                "additional_answers": {
                    "0": datasets.features.Sequence(
                        {
                            "span_start": datasets.Value("int32"),
                            "span_end": datasets.Value("int32"),
                            "span_text": datasets.Value("string"),
                            "input_text": datasets.Value("string"),
                            "turn_id": datasets.Value("int32"),
                        }
                    ),
                    "1": datasets.features.Sequence(
                        {
                            "span_start": datasets.Value("int32"),
                            "span_end": datasets.Value("int32"),
                            "span_text": datasets.Value("string"),
                            "input_text": datasets.Value("string"),
                            "turn_id": datasets.Value("int32"),
                        }
                    ),
                    "2": datasets.features.Sequence(
                        {
                            "span_start": datasets.Value("int32"),
                            "span_end": datasets.Value("int32"),
                            "span_text": datasets.Value("string"),
                            "input_text": datasets.Value("string"),
                            "turn_id": datasets.Value("int32"),
                        }
                    ),
                },
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
        data_dirs = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dirs["train"],
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dirs["validation"],
                    "split": datasets.Split.VALIDATION,
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for row in data["data"]:
                id = row["id"]
                source = row["source"]
                story = row["story"]
                questions = [
                    {"input_text": q["input_text"], "turn_id": q["turn_id"]}
                    for q in row["questions"]
                ]
                answers = [
                    {
                        "span_start": a["span_start"],
                        "span_end": a["span_end"],
                        "span_text": a["span_text"],
                        "input_text": a["input_text"],
                        "turn_id": a["turn_id"],
                    }
                    for a in row["answers"]
                ]
                if split == datasets.Split.TRAIN:
                    additional_answers = _EMPTY_ADDITIONAL_ANSWER
                else:
                    additional_answers = {
                        "0": [
                            {
                                "span_start": a0["span_start"],
                                "span_end": a0["span_end"],
                                "span_text": a0["span_text"],
                                "input_text": a0["input_text"],
                                "turn_id": a0["turn_id"],
                            }
                            for a0 in row["additional_answers"]["0"]
                        ],
                        "1": [
                            {
                                "span_start": a1["span_start"],
                                "span_end": a1["span_end"],
                                "span_text": a1["span_text"],
                                "input_text": a1["input_text"],
                                "turn_id": a1["turn_id"],
                            }
                            for a1 in row["additional_answers"]["1"]
                        ],
                        "2": [
                            {
                                "span_start": a2["span_start"],
                                "span_end": a2["span_end"],
                                "span_text": a2["span_text"],
                                "input_text": a2["input_text"],
                                "turn_id": a2["turn_id"],
                            }
                            for a2 in row["additional_answers"]["2"]
                        ],
                    }
                yield row["id"], {
                    "id": id,
                    "story": story,
                    "source": source,
                    "questions": questions,
                    "answers": answers,
                    "additional_answers": additional_answers,
                }
