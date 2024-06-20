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
"""SAT Analogy Questions dataset."""


import os

import datasets


_CITATION = """\
@article{article,
    author = {Turney, Peter},
    year = {2006},
    month = {09},
    pages = {379-416},
    title = {Similarity of Semantic Relations},
    volume = {32},
    journal = {Computational Linguistics},
    doi = {10.1162/coli.2006.32.3.379}
}
"""

_DESCRIPTION = """\
SAT (Scholastic Aptitude Test) Analogy Questions is a dataset comprising 374
multiple-choice analogy questions; 5 choices per question.
"""

_HOMEPAGE = "https://aclweb.org/aclwiki/SAT_Analogy_Questions_(State_of_the_art)"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""


class SatAnalogies(datasets.GeneratorBasedBuilder):
    """SAT (Scholastic Aptitude Test) Analogy Questions is a dataset comprising 374 multiple-choice analogy questions."""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="sat_analogies",
            version=VERSION,
            description="The SAT Analogy Questions dataset",
        ),
    ]

    @property
    def manual_download_instructions(self):
        return (
            "To use SAT Analogy Questions you have to download it manually. Please "
            "email Peter Turney to request the data (https://www.apperceptual.com). "
            "Once you receive a download link for the dataset, supply the local path "
            "as the `data_dir` arg: "
            "`datasets.load_dataset('sat_analogies', data_dir='path/to/folder/folder_name')`"
        )

    def _info(self):
        features = datasets.Features(
            {
                "source": datasets.Value("string"),
                "stem": datasets.Value("string"),
                "choices": datasets.features.Sequence(datasets.Value("string")),
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
        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"{data_dir} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('matinf', data_dir=...)` that includes SAT-package-V3.txt. Manual download instructions: {self.manual_download_instructions}"
            )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "SAT-package-V3.txt"),
                },
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath):
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            record = []
            for line in f:
                line = line.strip()
                if len(line) == 0 and record:
                    data.append(record)
                    record = []
                elif len(line) > 0 and line[0] == "#":
                    # Skip comments.
                    continue
                else:
                    record.append(line)
            data.append(record)
        for key, record in enumerate(data):
            source = record[-8]
            stem = record[-7]
            choices = record[-6:-1]
            solution = record[-1]
            yield key, {
                "source": source,
                "stem": stem,
                "choices": choices,
                "solution": solution,
            }
