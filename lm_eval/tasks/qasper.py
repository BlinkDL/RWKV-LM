"""
A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers
https://arxiv.org/abs/2105.03011

QASPER is a dataset of 5,049 questions over 1,585 Natural Language Processing papers.
Each question is written by an NLP practitioner who read only the title and abstract
of the corresponding paper, and the question seeks information present in the full
text. The questions are then answered by a separate set of NLP practitioners who also
provide supporting evidence to answers.

Homepage: https://allenai.org/data/qasper
"""
from collections import Counter
import re
import string
from lm_eval.base import rf, Task
from lm_eval.metrics import f1_score, mean


_CITATION = """
@article{DBLP:journals/corr/abs-2105-03011,
    author    = {Pradeep Dasigi and
               Kyle Lo and
               Iz Beltagy and
               Arman Cohan and
               Noah A. Smith and
               Matt Gardner},
    title     = {A Dataset of Information-Seeking Questions and Answers Anchored in
               Research Papers},
    journal   = {CoRR},
    volume    = {abs/2105.03011},
    year      = {2021},
    url       = {https://arxiv.org/abs/2105.03011},
    eprinttype = {arXiv},
    eprint    = {2105.03011},
    timestamp = {Fri, 14 May 2021 12:13:30 +0200},
    biburl    = {https://dblp.org/rec/journals/corr/abs-2105-03011.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def categorise_answer(answer_blob):
    if answer_blob["unanswerable"]:
        answer = "unanswerable"
        answer_type = "unanswerable"
        return answer, answer_type
    elif answer_blob["yes_no"]:
        answer = "yes"
        answer_type = "bool"
        return answer, answer_type
    elif answer_blob["free_form_answer"]:
        answer = answer_blob["free_form_answer"]
        answer_type = "free form answer"
        return answer, answer_type
    elif answer_blob["extractive_spans"]:
        answer = answer_blob["extractive_spans"]
        answer_type = "extractive_spans"
        return answer, answer_type
    elif answer_blob["yes_no"] is False:
        answer = "no"
        answer_type = "bool"
        return answer, answer_type


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


class QASPER(Task):
    VERSION = 0
    DATASET_PATH = "qasper"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def doc_to_text(self, doc):
        return (
            "TITLE: "
            + doc["title"]
            + "\n"
            + "ABSTRACT: "
            + doc["abstract"]
            + "\n\n"
            + "Q: "
            + doc["question"]
            + "\n\n"
            + "A:"
        )

    def doc_to_target(self, doc):
        answer = doc["answer"]
        if isinstance(answer, list):
            answer = ", ".join(answer)
        return " " + answer

    def training_docs(self):
        for doc in self.dataset["train"]:
            yield from self._process_doc(doc)

    def validation_docs(self):
        for doc in self.dataset["validation"]:
            yield from self._process_doc(doc)

    def _process_doc(self, doc):
        """Given a `doc`, flatten it out so that each JSON blob
        contains exactly one question and one answer. Logic taken from
        the reference implementation available at
        https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py
        """
        obs_list = []
        for question, answer_list in zip(doc["qas"]["question"], doc["qas"]["answers"]):
            for answer_blob in answer_list["answer"]:
                answer, answer_type = categorise_answer(answer_blob)
                obs_list.append(
                    {
                        "title": doc["title"],
                        "abstract": doc["abstract"],
                        "question": question,
                        "answer": answer,
                        "answer_type": answer_type,
                    }
                )
        return obs_list

    def process_results(self, doc, results):
        # TODO: Calculate a score for extractive spans once a request type for generating
        # extractive spans is available
        if not results:
            return {}
        elif len(results) == 1:
            [res] = results
        elif len(results) == 2:
            [ll_yes, ll_no] = results

        # TODO: Handle unanswerability first
        # unanswerable_gold = doc["answer_type"] == "unanswerable"
        # unanswerable_pred = exp(logprob_unanswerable)
        # res_dict["f1_unanswerable"] = (unanswerable_gold, unanswerable_pred)

        res_dict = {}
        # Handle yes/no questions
        if doc["answer_type"] == "bool":
            gold = 1 if doc["answer"] == "yes" else 0
            pred = ll_yes > ll_no
            res_dict["f1_yesno"] = (gold, pred)

        # Handle completions
        if doc["answer_type"] == "free form answer":
            res_dict["f1_abstractive"] = token_f1_score(res, doc["answer"])

        # TODO: Handle extraction
        # if doc["answer_type"] == "extractive_spans":
        #     res_dict["f1_extractive"] = 0
        return res_dict

    def aggregation(self):
        return {
            "f1_yesno": f1_score,
            "f1_abstractive": mean,
        }

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        # unanswerable = rf.loglikelihood(ctx, " " + "unanswerable")
        if doc["answer_type"] in ("free form answer"):
            return [rf.greedy_until(ctx, ["\n"])]
        elif doc["answer_type"] in ("bool"):
            ll_yes, _ = rf.loglikelihood(ctx, " yes")
            ll_no, _ = rf.loglikelihood(ctx, " no")
            return [ll_yes, ll_no]
        else:
            return []

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "f1_yesno": True,
            "f1_abstractive": True,
        }
