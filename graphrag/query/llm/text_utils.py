# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Text Utilities for LLM."""

import json
import logging
import re
from collections.abc import Iterator
from itertools import islice

import tiktoken
from json_repair import repair_json

import graphrag.config.defaults as defs

log = logging.getLogger(__name__)


def num_tokens(text: str, token_encoder: tiktoken.Encoding | None = None) -> int:
    """Return the number of tokens in the given text."""
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding(defs.ENCODING_MODEL)
    return len(token_encoder.encode(text))  # type: ignore


def batched(iterable: Iterator, n: int):
    """
    Batch data into tuples of length n. The last batch may be shorter.

    Taken from Python's cookbook: https://docs.python.org/3/library/itertools.html#itertools.batched
    """
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        value_error = "n must be at least one"
        raise ValueError(value_error)
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def chunk_text(
    text: str, max_tokens: int, token_encoder: tiktoken.Encoding | None = None
):
    """Chunk text by token length."""
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding(defs.ENCODING_MODEL)
    tokens = token_encoder.encode(text)  # type: ignore
    chunk_iterator = batched(iter(tokens), max_tokens)
    yield from (token_encoder.decode(list(chunk)) for chunk in chunk_iterator)


def try_parse_json_object(input: str, verbose: bool = True) -> tuple[str, dict]:
    """JSON cleaning and formatting utilities."""
    # Sometimes, the LLM returns a json string with some extra description, this function will clean it up.

    result = None
    try:
        # Try parse first
        result = json.loads(input)
    except json.JSONDecodeError:
        if verbose:
            log.debug("Warning: Error decoding faulty json, attempting repair")

    if result:
        return input, result

    # Try to find JSON object in the text
    pattern = r"\{(.*)\}"
    match = re.search(pattern, input, re.DOTALL)
    if match:
        input = "{" + match.group(1) + "}"
    else:
        if verbose:
            log.debug("No JSON object found in input (first 100 chars): %s", input[:100])
        return input, {}

    # Clean up json string.
    input = (
        input.replace("{{", "{")
        .replace("}}", "}")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", " ")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .strip()
    )

    # Remove JSON Markdown Frame
    if input.startswith("```json"):
        input = input[len("```json"):]
    if input.endswith("```"):
        input = input[:len(input) - len("```")]
    if input.startswith("```"):
        input = input[len("```"):]
    if input.endswith("```"):
        input = input[:len(input) - len("```")]

    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        if verbose:
            log.debug("Attempting to repair JSON: %s", input[:10])
        # Fixup potentially malformed json string using json_repair.
        try:
            input = str(repair_json(json_str=input, return_objects=False))
            result = json.loads(input)
        except (json.JSONDecodeError, Exception) as e:
            if verbose:
                log.debug("Failed to repair JSON: %s, error: %s", input[:100], str(e))
            return input, {}
    else:
        if not isinstance(result, dict):
            if verbose:
                log.debug("Parsed result is not a dictionary: %s", type(result))
            return input, {}
        return input, result

    return input, result
