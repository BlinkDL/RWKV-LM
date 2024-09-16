""" TextSynth API
Implementation provided by Fabrice Bellard:
    https://github.com/EleutherAI/lm-evaluation-harness/issues/295

In order to use the API, you must have a valid TextSynth account and
enough credits.

Example usage:

    python main.py --model textsynth --model_args engine=gptj_6B --no_cache --tasks piqa

Homepage: https://textsynth.com/index.html
"""
import logging
import os
import requests as _requests
import time
from tqdm import tqdm
from lm_eval.base import BaseLM


logger = logging.getLogger(__name__)


def textsynth_completion(**kwargs):
    """Query TextSynth API for completion.
    Retry with back-off until they respond.
    """
    backoff_time = 3
    while True:
        try:
            return _requests.post(**kwargs)
        except _requests.exceptions.RequestException:
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5


class TextSynthLM(BaseLM):
    def __init__(self, engine, truncate=False):
        """
        :param engine: str
            TextSynth API engine (e.g. `gptj_6B`)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()

        self.engine = engine
        self.truncate = truncate
        self.api_url = "https://api.textsynth.com"
        # Read from environment variable TEXTSYNTH_API_SECRET_KEY
        self.api_key = os.environ["TEXTSYNTH_API_SECRET_KEY"]

    @property
    def eot_token_id(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        raise NotImplementedError()

    @property
    def max_length(self):
        # NOTE: Turn on truncation to avoid errors on long inputs.
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        raise NotImplementedError()

    def tok_encode(self, string: str):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        raise NotImplementedError()

    def tok_decode(self, tokens):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        raise NotImplementedError()

    def loglikelihood(self, requests):
        res = []
        for context, continuation in tqdm(requests):
            response = textsynth_completion(
                url=self.api_url + "/v1/engines/" + self.engine + "/logprob",
                headers={"Authorization": "Bearer " + self.api_key},
                json={"context": context, "continuation": continuation},
            )
            resp = response.json()
            if "logprob" in resp:
                logprob = resp["logprob"]
                is_greedy = resp["is_greedy"]
                res.append((logprob, is_greedy))
            else:
                logger.error(
                    f"The following response does not contain `logprobs`. Got:\n{resp}"
                )
                assert False
        return res

    def loglikelihood_rolling(self, requests):
        # TODO: The TextSynth API does not support tokenized inputs so we cannot
        # manually partition long contexts into smaller rolling windows as
        # done for other models derived from `BaseLM`. Override this method
        # with a windowing scheme that works for direct string inputs.
        raise NotImplementedError(
            "`loglikelihood_rolling` is currently not supported due to lack of "
            "input tokenization support from TextSynth."
        )

    def greedy_until(self, requests):
        if not requests:
            return []

        res = []
        for request in tqdm(requests):
            inp = request[0]
            until = request[1]
            response = textsynth_completion(
                url=self.api_url + "/v1/engines/" + self.engine + "/completions",
                headers={"Authorization": "Bearer " + self.api_key},
                json={
                    "prompt": inp,
                    "max_tokens": self.max_gen_toks,
                    "top_k": 1,
                    "stop": until,
                },
            )
            resp = response.json()
            if "text" in resp:
                s = resp["text"]
                res.append(s)
            else:
                logger.error(
                    f"The following response does not contain generated `text`. "
                    "Got:\n{resp}"
                )
                assert False
        return res

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override greedy_until
        raise NotImplementedError()
