"""This module contains io specific utility functions."""
from __future__ import annotations

from collections import deque
from typing import Iterable


def glue_lines(lines: Iterable[str], glue_token: str = "\\") -> deque[str]:
    r"""Glue lines together based on a specific token.

    Example:
        >>> lines = ["statement 1", "this \\", "is \\", "statement 2", "statement 3"]
        >>> glue_lines(lines)
        deque(['statement 1', 'this is statement 2', 'statement 3'])

    Args:
        lines: Iterable of strings, possibly containing statements that need to be glued.
        glue_token: Token that specifies that two lines need to be glued together.

    Returns:
        A ``deque`` object containing the glued lines.
    """
    glued_lines = deque()
    glued_line = ""
    for line in lines:
        if line.endswith(glue_token):
            glued_line += line[:-1]
        else:
            glued_line += line
            glued_lines.append(glued_line)
            glued_line = ""

    return glued_lines


def split_lines(lines: Iterable[str], split_token: str = ";") -> deque[str]:
    r"""Split lines based on a specific token.

    Example:
        >>> lines = ["statement 1", "statement 2;statement 3", "statement 4"]
        >>> split_lines(lines)
        deque(['statement 1', 'statement 2', 'statement 3', 'statement 4'])

    Args:
        lines: Iterable of strings, possibly containing statements that need to be split.
        split_token: Token that specifies that a line need to be split.

    Returns:
        A ``deque`` object containing the split lines.
    """
    split_lines = deque()
    for line in lines:
        split_lines.extend(line.split(split_token))
    return split_lines


def remove_one_line_comments(lines: Iterable[str], token: str = "#") -> deque[str]:
    r"""Remove inline comments based on a starting token.

    Example:
        >>> lines = ["statement 1", "statement 2 # comment 1", "# comment 2"]
        >>> remove_one_line_comments(lines)
        deque(['statement 1', 'statement 2 '])

    Args:
        lines: Iterable of strings, possibly containing inline comments.
        token: Token that specifies inline comment.

    Returns:
        A ``deque`` object containing the lines without the inline comments.
    """
    new_lines = deque()
    for line in lines:
        if line.startswith(token):
            continue
        if token in line:
            new_lines.append(line[: line.index(token)])
            continue
        new_lines.append(line)

    return new_lines


def remove_block_comments(
    lines: Iterable[str], start_token: str = "/*", end_token: str = "*/"
) -> deque[str]:
    r"""Remove block comments, possibly spanning multiple lines.

    Example:
        >>> lines = ["statement /* comment 1", "still comment 1", "end */ 1"]
        >>> remove_block_comments(lines)
        deque(['statement  1'])

    Args:
        lines: Iterable of strings, possibly containing block comments.
        start_token: Token that specifies the start of the block comment.
        end_token: Token that specifies the end of the block comment.

    Returns:
        A ``deque`` object containing the lines without the block comments.
    """
    new_lines = deque()
    inside_block = False
    multi_line = ""
    lenght_end_token = len(end_token)
    for line in lines:
        if not inside_block:
            if line.startswith(start_token):
                inside_block = True
                continue
            if start_token in line:
                multi_line = line[: line.index(start_token)]
                inside_block = True
                continue
            new_lines.append(line)
        else:
            if line.endswith(end_token) and multi_line:
                new_lines.append(multi_line)
                multi_line = ""
                inside_block = False
            elif line.endswith(end_token):
                inside_block = False
            elif end_token in line:
                multi_line += line[line.index(end_token) + lenght_end_token :]
                new_lines.append(multi_line)
                multi_line = ""
                inside_block = False

    return new_lines


def strip_lines(lines: Iterable[str]) -> deque(str):
    """Strip lines of starting or trailing whitespaces.

    Example:
        >>> lines = ["  statement 1", "statement 2", "statement 3 \t "]
        >>> strip_lines(lines)
        deque(['statement 1', 'statement 2', 'statement 3'])

    Args:
        lines: Iterable of string, possibly containing trailing or starting whitespaces.

    Returns:
        A ``deque`` object containing the lines without starting or trailing whitespaces.
    """
    new_lines = deque()
    for line in lines:
        new_line = line.strip()
        if new_line:
            new_lines.append(new_line)
    return new_lines
