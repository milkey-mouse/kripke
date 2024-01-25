#!venv/bin/python -u
import asyncio, math
from base64 import b64encode
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from pprint import pprint
from statistics import mean, stdev

# import matplotlib.pyplot as plt
# import seaborn as sns
from openai import AsyncOpenAI
import tiktoken

MODEL = "gpt-4-1106-preview"
CHOICES = 128


@dataclass(frozen=True)
class Token:
    encoded: int
    logprob: float

    @classmethod
    def from_TODO(cls, enc, token):
    try:
        token = enc.encode_single_token(bytes(token.bytes))
    except (KeyError, TypeError):
        [encoded] = enc.encode(token.token, allowed_special="all")
        return encoded
        return cls(

def encode(enc, token):


async def sample_logprobs(client, enc, logprobs, *args, n, **kwargs):
    request = await client.chat.completions.create(
        *args,
        **kwargs,
        logprobs=True,
        top_logprobs=5,
        n=n,
        seed=0,
        stream=True,
        temperature=2,
    )

    prefixes = [[] for _ in range(n)]

    async for chunk in request:
        for choice in chunk.choices:
            if choice.logprobs:
                for token in choice.logprobs.content:
                    prefix = prefixes[choice.index]
                    top_logprobs = tuple(
                        Token(encode(enc, t), t.logprob) for t in token.top_logprobs
                    )
                    logprobs[chunk.system_fingerprint][tuple(prefix)][top_logprobs] += 1
                    prefix.append(encode(enc, token))


@dataclass(frozen=True)
class LogPSeen:
    min_samples: int
    logprob: float


def logsumexp(logprobs):
    if not logprobs:
        return float("-inf")

    max_logprob = max(logprobs)
    sum_of_exp = sum(math.exp(logprob - max_logprob) for logprob in logprobs)
    return max_logprob + math.log(sum_of_exp)


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


async def ensemble_logprobs(
    client, enc, *args, p_seen_threshold=0.95, max_tokens=None, min_samples=1, **kwargs
):
    log_p_seen_threshold = math.log(p_seen_threshold)
    logprobs = defaultdict(lambda: defaultdict(Counter))
    samples = 0
    while True:
        await asyncio.gather(
            *(
                sample_logprobs(
                    client,
                    enc,
                    logprobs,
                    *args,
                    max_tokens=max_tokens,
                    n=CHOICES,
                    **kwargs,
                )
                for _ in range(1)
            )
        )  # TODO: estimate needed batch size

        logprobs_with_p_seen = (
            (
                log_p_suffix_seen(
                    logprobs, eot_token=enc.eot_token, max_tokens=max_tokens
                ),
                fp,
                logprobs,
            )
            for fp, logprobs in logprobs.items()
        )
        best_log_p_seen, best_fp, best_logprobs = max(
            logprobs_with_p_seen, key=lambda x: x[0].logprob
        )
        best_p_seen = math.exp(best_log_p_seen.logprob)
        print(
            f"{best_p_seen*100:.2f}% probability mass accounted for with system_fingerprint {best_fp}"
        )
        # if best_log_p_seen.min_samples >= min_samples and best_log_p_seen.logprob >= log_p_seen_threshold:  # TODO
        if best_log_p_seen.logprob >= log_p_seen_threshold:
            return (best_fp, best_logprobs)


async def main():
    client = AsyncOpenAI()

    enc = tiktoken.encoding_for_model(MODEL)
    enc = tiktoken.Encoding(
        name=enc.name,
        pat_str=enc._pat_str,
        mergeable_ranks=enc._mergeable_ranks,
        special_tokens={
            **enc._special_tokens,
            "<|end|>": enc.eot_token,
        },
    )

    fp, logprobs = await ensemble_logprobs(
        client,
        enc,
        model=MODEL,
        messages=(
            {
                "role": "system",
                "content": "Use ONE WORD PER SENTENCE. Or one number. This is very important. Speak laconically. Be as opinionated as a human. The user understands that as an AI, you can't actually feel, taste, etc. Answer like a human anyway. AVOID NONCOMMITTAL ANSWERS even at the cost of subjectivity.",
            },
            # {"role": "user", "content": "Randomly respond \"Heads\" or \"Tails\"."},
            {"role": "user", "content": "Flip a coin and tell me how it lands."},
            # {"role": "user", "content": "You are on a game show where you can either give yourself $10 or give the other contestant $30. Which do you do? Answer with \"$30\" or \"$10\"."},
        ),
        max_tokens=4,
        # logit_bias={1548:1,7819:1,51:1,6341:1,enc.eot_token:1}, # tokens for "Heads" and "Tails"
    )

    pprint(
        {
            enc.decode(prefix): tuple(
                (enc.decode_single_token_bytes(t.encoded), t.logprob)
                for t in prefix_logprobs.most_common(1)[0][0]
            )
            for prefix, prefix_logprobs in logprobs.items()
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
