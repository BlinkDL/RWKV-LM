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
# Custom DROP dataset that, unlike HF, keeps all question-answer pairs
# even if there are multiple types of answers for the same question.
"""DROP dataset."""


import json
import os

import datasets


_CITATION = """\
@misc{dua2019drop,
    title={DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs},
    author={Dheeru Dua and Yizhong Wang and Pradeep Dasigi and Gabriel Stanovsky and Sameer Singh and Matt Gardner},
    year={2019},
    eprint={1903.00161},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
DROP is a QA dataset which tests comprehensive understanding of paragraphs. In
this crowdsourced, adversarially-created, 96k question-answering benchmark, a
system must resolve multiple references in a question, map them onto a paragraph,
and perform discrete operations over them (such as addition, counting, or sorting).
"""

_HOMEPAGE = "https://allenai.org/data/drop"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URLS = {
    "drop": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip",
}

_EMPTY_VALIDATED_ANSWER = [
    {
        "number": "",
        "date": {
            "day": "",
            "month": "",
            "year": "",
        },
        "spans": [],
        "worker_id": "",
        "hit_id": "",
    }
]


class Drop(datasets.GeneratorBasedBuilder):
    """DROP is a QA dataset which tests comprehensive understanding of paragraphs."""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="drop", version=VERSION, description="The DROP dataset."
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "section_id": datasets.Value("string"),
                "passage": datasets.Value("string"),
                "question": datasets.Value("string"),
                "query_id": datasets.Value("string"),
                "answer": {
                    "number": datasets.Value("string"),
                    "date": {
                        "day": datasets.Value("string"),
                        "month": datasets.Value("string"),
                        "year": datasets.Value("string"),
                    },
                    "spans": datasets.features.Sequence(datasets.Value("string")),
                    "worker_id": datasets.Value("string"),
                    "hit_id": datasets.Value("string"),
                },
                "validated_answers": datasets.features.Sequence(
                    {
                        "number": datasets.Value("string"),
                        "date": {
                            "day": datasets.Value("string"),
                            "month": datasets.Value("string"),
                            "year": datasets.Value("string"),
                        },
                        "spans": datasets.features.Sequence(datasets.Value("string")),
                        "worker_id": datasets.Value("string"),
                        "hit_id": datasets.Value("string"),
                    }
                ),
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
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "drop_dataset", "drop_dataset_train.json"
                    ),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "drop_dataset", "drop_dataset_dev.json"
                    ),
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            key = 0
            for section_id, example in data.items():
                # Each example (passage) has multiple sub-question-answer pairs.
                for qa in example["qa_pairs"]:
                    # Build answer.
                    answer = qa["answer"]
                    answer = {
                        "number": answer["number"],
                        "date": {
                            "day": answer["date"].get("day", ""),
                            "month": answer["date"].get("month", ""),
                            "year": answer["date"].get("year", ""),
                        },
                        "spans": answer["spans"],
                        "worker_id": answer.get("worker_id", ""),
                        "hit_id": answer.get("hit_id", ""),
                    }
                    validated_answers = []
                    if "validated_answers" in qa:
                        for validated_answer in qa["validated_answers"]:
                            va = {
                                "number": validated_answer.get("number", ""),
                                "date": {
                                    "day": validated_answer["date"].get("day", ""),
                                    "month": validated_answer["date"].get("month", ""),
                                    "year": validated_answer["date"].get("year", ""),
                                },
                                "spans": validated_answer.get("spans", ""),
                                "worker_id": validated_answer.get("worker_id", ""),
                                "hit_id": validated_answer.get("hit_id", ""),
                            }
                            validated_answers.append(va)
                    else:
                        validated_answers = _EMPTY_VALIDATED_ANSWER
                    yield key, {
                        "section_id": section_id,
                        "passage": example["passage"],
                        "question": qa["question"],
                        "query_id": qa["query_id"],
                        "answer": answer,
                        "validated_answers": validated_answers,
                    }
                    key += 1
