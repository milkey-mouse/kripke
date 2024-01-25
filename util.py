#!venv/bin/python
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
import tiktoken

# Kripke: GPT completions from all possible worlds

import asyncio, math, os, string, sys, types
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import chain, repeat, zip_longest
from pprint import pprint
from typing import Optional
from math import log, exp

def maybe_decode(enc, token):
    try:
        return ENCODER.decode_single_token_bytes(token).decode("utf-8")
    except:
        return token

def encodings(s, prefix=[], suffix=[]):
    if not s:
        yield [*prefix, *suffix]
    else:
        try:
            for i in range(1, len(s)+1):
                new_prefix = [*prefix, enc.encode_single_token(s[:i])]
                yield from encodings(s[i:], new_prefix, suffix)
        except KeyError:
            pass

def logsumexp(logprobs):
    if not logprobs:
        return float("-inf")

    max_logprob = max(logprobs)
    sum_of_exp = sum(math.exp(logprob - max_logprob) for logprob in logprobs)
    return max_logprob + math.log(sum_of_exp)

def encode(enc, token):
    try:
        return enc.encode_single_token(bytes(token.bytes))
    except (KeyError, TypeError):
        [encoded] = enc.encode(token.token, allowed_special="all")
        return encoded

@dataclass(frozen=True)
class Token:
    logprobs: dict[str, float] = field(default_factory=dict)
    logit_bias: dict[int, float] = field(default_factory=dict)

@dataclass(frozen=True)
class Token:
    encoded: int
    logprob: float

def async_agnostic(corofn):
    @wraps(corofn)
    def wrapper(*args, **kwargs):
        with suppress(RuntimeError):
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return corofn(*args, **kwargs)
        return asyncio.run(corofn(*args, **kwargs))

    return wrapper

@dataclass(frozen=True)
class LogPSeen:
    min_samples: int
    logprob: float

def log_p_suffix_seen(prefixes, prefix=[], eot_token=None, max_tokens=None):
    if prefix[-1:] == (eot_token,) or len(prefix) == max_tokens:
        return LogPSeen(float("inf"), 0)

    key = tuple(prefix)
    if key not in prefixes:
        return LogPSeen(0, float("-inf"))
    else:
        suffix_logprobs, min_samples = prefixes[key].most_common(1)[0]
        del key

    logprobs = []
    for token in suffix_logprobs:
        prefix.append(token.encoded)
        prefix_with_token = log_p_suffix_seen(prefixes, prefix, eot_token, max_tokens)
        prefix.pop()

        min_samples = min(min_samples, prefix_with_token.min_samples)
        token_logprob_sans_prefix = prefix_with_token.logprob + token.logprob
        if token_logprob_sans_prefix != float("-inf"):
            logprobs.append(token_logprob_sans_prefix)

    return LogPSeen(min_samples, logsumexp(logprobs))
