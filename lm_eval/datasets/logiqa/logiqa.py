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
"""LogiQA dataset."""


import datasets


_CITATION = """\
@misc{liu2020logiqa,
    title={LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning},
    author={Jian Liu and Leyang Cui and Hanmeng Liu and Dandan Huang and Yile Wang and Yue Zhang},
    year={2020},
    eprint={2007.08124},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
LogiQA is a dataset for testing human logical reasoning. It consists of 8,678 QA
instances, covering multiple types of deductive reasoning. Results show that state-
of-the-art neural models perform by far worse than human ceiling. The dataset can
also serve as a benchmark for reinvestigating logical AI under the deep learning
NLP setting.
"""

_HOMEPAGE = "https://github.com/lgw863/LogiQA-dataset"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

_URLS = {
    "train": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Train.txt",
    "validation": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Eval.txt",
    "test": "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt",
}


class Logiqa(datasets.GeneratorBasedBuilder):
    """LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning"""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="logiqa", version=VERSION, description="The LogiQA dataset."
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "label": datasets.Value("string"),
                "context": datasets.Value("string"),
                "question": datasets.Value("string"),
                "options": datasets.features.Sequence(datasets.Value("string")),
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
        urls = {
            "train": _URLS["train"],
            "test": _URLS["test"],
            "validation": _URLS["validation"],
        }
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
        def normalize(text):
            return text.replace(".", ". ").strip()

        with open(filepath, encoding="utf-8") as f:
            data = f.read().strip().split("\n\n")
            for key, row in enumerate(data):
                example = row.split("\n")
                yield key, {
                    "label": example[0].strip(),
                    "context": normalize(example[1]),
                    "question": normalize(example[2]),
                    "options": [normalize(option[2:]) for option in example[3:]],
                }
