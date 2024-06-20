"""
LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning
https://arxiv.org/pdf/2007.08124.pdf

LogiQA is a dataset for testing human logical reasoning. It consists of 8,678 QA
instances, covering multiple types of deductive reasoning. Results show that state-
of-the-art neural models perform by far worse than human ceiling. The dataset can
also serve as a benchmark for reinvestigating logical AI under the deep learning
NLP setting.

Homepage: https://github.com/lgw863/LogiQA-dataset
"""
import inspect
import lm_eval.datasets.logiqa.logiqa
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@misc{liu2020logiqa,
    title={LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning},
    author={Jian Liu and Leyang Cui and Hanmeng Liu and Dandan Huang and Yile Wang and Yue Zhang},
    year={2020},
    eprint={2007.08124},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


class LogiQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = inspect.getfile(lm_eval.datasets.logiqa.logiqa)
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        def format_example(doc, choices):
            """
            Passage: <passage>
            Question: <question>
            Choices:
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Answer:
            """
            prompt = "Passage: " + doc["context"] + "\n"
            prompt += "Question: " + doc["question"] + "\nChoices:\n"
            for choice, option in zip(choices, doc["options"]):
                prompt += f"{choice.upper()}. {option}\n"
            prompt += "Answer:"
            return prompt

        choices = ["a", "b", "c", "d"]
        return {
            "passage": doc["context"],  # Used for decontamination
            "query": format_example(doc, choices),
            "choices": doc["options"],
            "gold": choices.index(doc["label"]),
        }

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["passage"]
