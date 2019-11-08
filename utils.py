import codecs
from collections import namedtuple
import io
import re
import sys

import lark

NonTerminal = namedtuple('NonTerminal', ['name'])
Terminal = namedtuple('Terminal', ['name'])


def pretty(tree, stream=None):
    if stream is None:
        stream = io.StringIO()
    pretty_stream(tree, stream, 0)
    return stream.getvalue()


def pretty_stream(tree, stream, depth=0):
    indentation = '| ' * depth
    if isinstance(tree, lark.Tree):
        print(indentation + tree.data, file=stream)
        for t in tree.children:
            pretty_stream(t, stream, depth + 1)
    else:
        print(indentation + '%s %s' % (tree.type, repr(tree.value)), file=stream)


def _recurse_tree(tree, func):
    return list(map(func, tree.children)) if isinstance(tree, lark.Tree) else tree


def flatten_unnecessary_nodes(tree):
    new = _recurse_tree(tree, flatten_unnecessary_nodes)
    if isinstance(tree, lark.Tree):
        tree.children = new
        if len(new) == 1 and isinstance(new[0], lark.Tree):
            tree = new[0]
    return tree


def print_rules_dict(rules_dict, stream=sys.stdout):
    indentation = ' ' * 4
    for rule, expression in rules_dict.items():
        print(str(rule) + ':', file=stream)
        for sequence in expression:
            print(indentation + str(sequence), file=stream)


ESCAPE_SEQUENCE_RE = re.compile(r'''
    ( \\U........      # 8-digit hex escapes
    | \\u....          # 4-digit hex escapes
    | \\x..            # 2-digit hex escapes
    | \\[0-7]{1,3}     # Octal escapes
    | \\N\{[^}]+\}     # Unicode characters by name
    | \\[\\'"abfnrtv]  # Single-character escapes
    )''', re.UNICODE | re.VERBOSE)


def decode_escapes(s):
    def decode_match(match):
        return codecs.decode(match.group(0), 'unicode-escape')

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)


def visit(non_terminal, rules_dict):
    # rules_dict: {NonTerminal : expression}
    reachable_set = set()
    visit_stack = [non_terminal]
    while len(visit_stack) > 0:
        non_terminal = visit_stack.pop()
        reachable_set.add(non_terminal)
        for sequence in rules_dict[non_terminal]:
            for symbol in sequence:
                if isinstance(symbol, NonTerminal) and symbol not in reachable_set:
                    visit_stack.append(symbol)
    return reachable_set


class Vocabulary:
    def __init__(self, oov_token):
        self._unused_value = 0
        self._token_to_value = {}
        self._value_to_token = {}
        self.oov_token = oov_token
        self.add_token(oov_token)

    def add_token(self, token):
        if token not in self._token_to_value:
            self._token_to_value[token] = self._unused_value
            self._value_to_token[self._unused_value] = token
            self._unused_value += 1

    def value_of_token(self, token):
        if token not in self._token_to_value:
            return self._token_to_value[self.oov_token]
        return self._token_to_value[token]

    def token_of_value(self, value):
        if value not in self._value_to_token:
            return self.oov_token
        return self._value_to_token[value]

    def vocab_size(self):
        return len(self._token_to_value)
