from __future__ import annotations

from collections import deque

import pytest

from qgym.io.utils import (
    glue_lines,
    remove_block_comments,
    remove_one_line_comments,
    split_lines,
    strip_lines,
)


@pytest.mark.parametrize(
    "input_lines, glue_token, expected_output",
    [
        (
            ["statement 1", "this \\", "is \\", "statement 2", "statement 3"],
            "\\",
            deque(["statement 1", "this is statement 2", "statement 3"]),
        ),
        (
            ["statement 1", "this \\", "is \\", "statement 2", "statement 3"],
            "0",
            deque(["statement 1", "this \\", "is \\", "statement 2", "statement 3"]),
        ),
    ],
)
def test_glue_lines(
    input_lines: list[str], glue_token: str, expected_output: deque[str]
) -> None:
    assert glue_lines(input_lines, glue_token) == expected_output


@pytest.mark.parametrize(
    "input_lines, split_token, expected_output",
    [
        (
            ["statement 1", "statement 2;statement 3", "statement 4"],
            ";",
            deque(["statement 1", "statement 2", "statement 3", "statement 4"]),
        ),
        (
            ["statement 1", "statement 2;statement 3", "statement 4"],
            "0",
            deque(
                ["statement 1", "statement 2;statement 3", "statement 4"],
            ),
        ),
    ],
)
def test_split_lines(
    input_lines: list[str], split_token: str, expected_output: deque[str]
) -> None:
    assert split_lines(input_lines, split_token) == expected_output


@pytest.mark.parametrize(
    "input_lines, token, expected_output",
    [
        (
            ["statement 1", "statement 2 # comment 1", "# comment 2"],
            "#",
            deque(["statement 1", "statement 2 "]),
        ),
        (
            ["statement 1", "statement 2 # comment 1", "# comment 2"],
            "0",
            deque(["statement 1", "statement 2 # comment 1", "# comment 2"]),
        ),
    ],
)
def test_remove_one_line_comments(
    input_lines: list[str], token: str, expected_output: deque[str]
) -> None:
    assert remove_one_line_comments(input_lines, token) == expected_output


@pytest.mark.parametrize(
    "input_lines, split_token, expected_output",
    [
        (
            ["statement 1", "statement 2;statement 3", "statement 4"],
            ";",
            deque(["statement 1", "statement 2", "statement 3", "statement 4"]),
        ),
        (
            ["statement 1", "statement 2;statement 3", "statement 4"],
            "0",
            deque(
                ["statement 1", "statement 2;statement 3", "statement 4"],
            ),
        ),
    ],
)
def test_split_lines(
    input_lines: list[str], split_token: str, expected_output: deque[str]
) -> None:
    assert split_lines(input_lines, split_token) == expected_output


@pytest.mark.parametrize(
    "input_lines, start_token, end_token, expected_output",
    [
        (
            ["statement /* comment 1", "still comment 1", "end */ 1"],
            "/*",
            "*/",
            deque(["statement  1"]),
        ),
        (
            ["statement /* comment 1", "still comment 1", "end */ 1"],
            "0",
            "0",
            deque(["statement /* comment 1", "still comment 1", "end */ 1"]),
        ),
    ],
)
def test_remove_block_comments(
    input_lines: list[str],
    start_token: str,
    end_token: str,
    expected_output: deque[str],
) -> None:
    assert remove_block_comments(input_lines, start_token, end_token) == expected_output


def test_strip_lines() -> None:
    lines = ["  statement 1", "statement 2", "statement 3 \t "]
    expected_output = deque(["statement 1", "statement 2", "statement 3"])
    assert strip_lines(lines) == expected_output
