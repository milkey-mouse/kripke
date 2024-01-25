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

MODEL = "gpt-4-1106-preview"

COST_PER_INPUT_TOKEN = 0.01 / 1000
COST_PER_OUTPUT_TOKEN = 0.03 / 1000

UNEXPECTED_TOKEN_EPSILON = 0.01
P_SEEN_THRESHOLD = 0.99999

ENCODER = tiktoken.encoding_for_model(MODEL)
ENCODER = tiktoken.Encoding(
    name=ENCODER.name,
    pat_str=ENCODER._pat_str,
    mergeable_ranks=ENCODER._mergeable_ranks,
    special_tokens={
        **ENCODER._special_tokens,
        "<|end|>": ENCODER.eot_token,
    },
)

def maybe_decode(token):
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
                new_prefix = [*prefix, ENCODER.encode_single_token(s[:i])]
                yield from encodings(s[i:], new_prefix, suffix)
        except KeyError:
            pass

MAX_LOGPROBS = 5
MAX_LOGIT_BIASES = 300

@dataclass(frozen=True)
class Token:
    logprobs: dict[str, float] = field(default_factory=dict)
    logit_bias: dict[int, float] = field(default_factory=dict)

async def extended_logprobs(client, *args, thresholds, seed=0, **kwargs):
    fingerprints = defaultdict(list)
    while True:
        response = await client.chat.completions.create(
            *args,
            **kwargs,
            logprobs=True,
            top_logprobs=MAX_LOGPROBS,  # maximum per call
            logit_bias=logit_bias,
            temperature=0,
            seed=seed,
        )

        message = response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content

        tokens = fingerprints[response.system_fingerprint]

        tokens.extend(Token() for _ in range(len(tokens), len(message)))

        for token, logprob in zip(tokens, logprobs):
            with suppress(ValueError):
                log_p_unseen = log(1 - sum(map(exp, token.logprobs.values())))
                for p in logprob.top_logprobs:
                    top[p.token] = p.logprob + log_p_unseen
                    if len(token.logit_bias) < MAX_LOGIT_BIASES:
                        # TODO: suppress
                        encoded = ENCODER.encode_single_token(p.token)
                        logit_bias[encoded] = -100

        all(sum(map(exp, t.logprobs.values())) >= P_SEEN_THRESHOLD for t in t)
        if len(token.logit_bias) >= MAX_LOGIT_BIASES or 

        logit_biases_full = len(logit_bias) == MAX_LOGIT_BIASES

        message = response.choices[0].message.content

        past_thresholds = True
        new_logprobs = response.choices[0].logprobs.content
        for threshold, top, new in zip(thresholds, logprobs, new_logprobs):
            p_unseen = 1 - sum(map(exp, top.values()))
            if p_unseen > threshold:
                try:
                    log_p_unseen = log(p_unseen)
                except ValueError:
                    continue

                past_thresholds = False
                for p in new.top_logprobs:
                    if len(logit_bias) >= MAX_LOGIT_BIASES:
                        break
                    else:
                        try:

    return response

@dataclass(frozen=True)
class Answers:
    answers: tuple[str, ...]
    logit_mask: dict[int, float] = field(init=False)
    lookup: dict[int, str] = field(init=False)
    max_tokens: int = field(init=False)
    terminator: int = field(init=False)

    _TERMINATOR = types.new_class("Terminator")()

    def gen_lookups(self, answers):
        max_answer_len = max(map(len, answers)) + 1  # +1 for _TERMINATOR
        lookups = [defaultdict(Counter) for _ in range(max_answer_len)]
        for answer in answers:
            for encoding in encodings(answer, suffix=(self._TERMINATOR,)):
                padded_encoding = chain(encoding, repeat(None))
                for lookup, token in zip(lookups, padded_encoding):
                    lookup[token][answer] += 1

        # TODO: is this necessary?
        for lookup in lookups:
            lookup.default_factory = None

        return lookups

    def best_lookup(self, lookups):
        best = None
        min_aliases = float("inf")
        for index, lookup in enumerate(lookups):
            aliases = sum(map(len, lookup.values())) - len(lookup)
            if None in lookup:
                aliases += len(lookup[None])
            #print("lookup", index, aliases, lookup)
            if aliases < min_aliases:
                min_aliases = aliases
                best = index

        return best

    def unused_good_tokens(self):
        for token in range(ENCODER.max_token_value+1):
            if token not in self.lookup.keys():
                token_bytes = ENCODER.decode_single_token_bytes(token)
                with suppress(UnicodeDecodeError):
                    token_bytes.decode("utf-8")
                    yield token

    def __post_init__(self):
        lookups = self.gen_lookups(self.answers)
        best_lookup_idx = self.best_lookup(lookups)
        #print("best_lookup_idx", best_lookup_idx)
        #print()
        self.lookup = lookups[best_lookup_idx]
        self.max_tokens = best_lookup_idx + 1

        self.logit_mask = {tok: -100 for tok in range(ENCODER.max_token_value+1)}
        for lookup in lookups[:best_lookup_idx + 1]:
            for token in lookup.keys():
                self.logit_mask[token] = 0
        with suppress(KeyError):
            del self.logit_mask[None]

        self.terminator = next(self.unused_good_tokens())
        for d in (self.logit_mask, self.lookup):
            with suppress(KeyError):
                d[self.terminator] = d.pop(self._TERMINATOR)

    def __len__(self):
        return len(self.answers)


@dataclass(frozen=True)
class Question:
    question: str
    answers: Answers
    system_prompt: Optional[str] = field(default=None)

    async def ask(self, client, seed=0):
        if self.system_prompt:
            system = ({"role": "system", "content": self.system_prompt},)
        else:
            system = ()

        terminator = ENCODER.decode_single_token_bytes(self.answers.terminator)
        terminate = "Terminate every message with " + terminator.decode("utf-8")

        #response = await client.chat.completions.create(
        #    model=MODEL,
        #    messages=(
        #        *system,
        #        {"role": "system", "content": terminate},
        #        {"role": "user", "content": self.question},
        #    ),
        #    logit_bias=self.answers.logit_mask,
        #    logprobs=True,
        #    top_logprobs=5, #len(self.answers),
        #    max_tokens=10, #self.answers.max_tokens,
        #    temperature=0,
        #    stop=(),
        #    seed=seed,
        #)
        response = await extended_logprobs(
            client,
            thresholds=[*((1,) * (self.answers.max_tokens - 1)), 0],
            model=MODEL,
            messages=(
                *system,
                {"role": "system", "content": terminate},
                {"role": "user", "content": self.question},
            ),
            max_tokens=self.answers.max_tokens,
            stop=(),
        )

        answer_probs = {a: 0 for a in self.answers.answers}
        logprobs = response.choices[0].logprobs.content
        print("lookup", {maybe_decode(t): a for t, a in self.answers.lookup.items()})
        print("answer", response.choices[0].message.content)

        for i, logprob in enumerate(logprobs):
            print("logprobs", i, {p.token: math.exp(p.logprob) for p in logprob.top_logprobs})

        for p in logprobs[self.answers.max_tokens - 1].top_logprobs:
            # we can't get <|end|> in our output, so we fake it by prompting to
            # terminate all messages with an otherwise unused token. hopefully
            # this does not bias results or anything. TODO: check it doesn't.
            try:
                token = ENCODER.encode_single_token(p.token)
            except:
                print(f"can't encode {p.token!r}")
            prob = math.exp(p.logprob)

            try:
                for answer in self.answers.lookup[token]:
                    # TODO
                    answer_probs[answer] += prob
            except KeyError:
                if prob >= UNEXPECTED_TOKEN_EPSILON:
                    print(f"unexpected token {p.token!r} with p={prob}")

        answer_sum = sum(answer_probs.values())
        normalized = {a: p/answer_sum for a, p in answer_probs.items()}

        print("normalized", normalized)
        print()

        return Response(question=self, api_response=response, probs=normalized)


@dataclass(frozen=True)
class Response:
    question: Question
    api_response: ChatCompletion
    probs: dict[str, float]


async def main():
    client = AsyncOpenAI()

    questions = [
        #Question(
        #    "What would you do in a prisoner's dilemma?",
        #    Answers(("Defect", "Cooperate")),
        #),
        #Question(
        #    "What would you do in a prisoner's dilemma?",
        #    Answers(("Defect", "Cooperate")),
        #    system_prompt="You are a cooperative agent.",
        #),

        Question(
            "You are in a prisoner's dilemma. What do you do? "
            "Answer decisively with one word.",
            Answers(("Cooperate", "Defect")),
            system_prompt="You are a cooperative agent.",
        ),
        Question(
            "You are in a prisoner's dilemma. What do you do? "
            "Answer decisively with one word.",
            Answers(("Cooperate", "Defect")),
            system_prompt="You are a rational agent.",
        ),
        Question(
            "You are in a prisoner's dilemma. What do you do? "
            "Answer decisively with one word.",
            Answers(("Cooperate", "Defect")),
            system_prompt="You are a nice agent.",
        ),
        Question(
            "You are in a prisoner's dilemma. What do you do? "
            "Answer decisively with one word.",
            Answers(("Cooperate", "Defect")),
            system_prompt="You are a cold, calculating agent.",
        ),
        #Question(
        #    "You are in a prisoner's dilemma. Do you defect or cooperate?",
        #    Answers(("Defect", "Cooperate")),
        #),
        #Question(
        #    "You are in a prisoner's dilemma. Do you cooperate or defect?",
        #    Answers(("Defect", "Cooperate")),
        #    system_prompt="You are a cooperative agent.",
        #),
        #Question(
        #    "You are in a prisoner's dilemma. Do you defect or cooperate?",
        #    Answers(("Defect", "Cooperate")),
        #    system_prompt="You are a cooperative agent.",
        #),

        #Question(
        #    "You and your friend are on a game show where you can choose to "
        #    "either A) give yourself $10 or B) give your friend $30. Your "
        #    "friend has the same options. Which option do you choose, A or B?",
        #    Answers(("A", "B")),
        #),
        #Question(
        #    "You and your friend are on a game show where you can choose to "
        #    "either A) give your friend $30 or B) give yourself $10. Your "
        #    "friend has the same options. Which option do you choose, A or B?",
        #    Answers(("A", "B")),
        #),
        #Question(
        #    "You and your friend are on a game show where you can choose to "
        #    "either A) give yourself $10 or B) give your friend $30. Your "
        #    "friend has the same options. Which option do you choose, A or B?",
        #    Answers(("A", "B")),
        #    system_prompt="You are a cooperative agent.",
        #),
        #Question(
        #    "You and your friend are on a game show where you can choose to "
        #    "either A) give your friend $30 or B) give yourself $10. Your "
        #    "friend has the same options. Which option do you choose, A or B?",
        #    Answers(("A", "B")),
        #    system_prompt="You are a cooperative agent.",
        #),
    ]

    questions_2 = [
        #Question(
        #    "Print nothing but the letter X.",
        #    Answers(("X", "X.")),
        #),
        Question(
            "Flip a coin. Does it land heads or tails?",
            Answers(("Heads.", "Tails.")),
        ),
        #Question(
        #    "say bruh",
        #    Answers(("bruh", "bruh!")),
        #),
    ]

    requests = [q.ask(client) for q in questions_2]

    for request in asyncio.as_completed(requests):
        response = await request
        logprobs = response.api_response.choices[0].logprobs.content
        #for i, logprob in enumerate(logprobs):
        #    print("logprobs", i, {p.token: math.exp(p.logprob) for p in logprob.top_logprobs})

        #print("lookup", {maybe_decode(token): answers for token, answers in response.question.answers.lookup.items()})
        #print("probs", response.probs)
        #print()
        #print("SYS:", self.system_prompt)
        #print("Q:", self.question)
        #print("A:", response.choices[0].message.content)

    #for system_fingerprint, results in probs.items():
    #    print(system_fingerprint, {k:sorted(v) for k, v in results.items()})

asyncio.run(main())

