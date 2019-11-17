import codecs
from collections import namedtuple
import io
import os
import re
import sys

import lark

NonTerminal = namedtuple('NonTerminal', ['name'])
Terminal = namedtuple('Terminal', ['name'])


class SimpleTree:
    def __init__(self, data="", children=None, used_rule=()):
        if children is None:
            children = []
        self.data = data
        self.children = children
        self.used_rule = used_rule

    @classmethod
    def from_lark_tree(cls, tree):
        new = recurse_tree(tree, SimpleTree.from_lark_tree)
        used_rule = []
        if new:
            for symbol in new:
                symbol = NonTerminal(symbol.data) if isinstance(symbol, SimpleTree) else Terminal(symbol.value)
                used_rule.append(symbol)
        else:
            used_rule.append(NonTerminal(""))
        return cls(tree.data, new, tuple(used_rule))

    def __repr__(self):
        return 'Tree(%s, %s)' % (self.data, self.children)


def pretty(tree, stream=None):
    if stream is None:
        stream = io.StringIO()
    pretty_stream(tree, stream, 0)
    return stream.getvalue()


def pretty_stream(tree, stream, depth=0):
    indentation = '| ' * depth
    if hasattr(tree, 'data') and hasattr(tree, 'children'):
        print(indentation + tree.data, file=stream)
        for t in tree.children:
            pretty_stream(t, stream, depth + 1)
    else:
        print(indentation + '%s %s' % (tree.type, repr(tree.value)), file=stream)


def recurse_tree(tree, func, recurse_tokens=False):
    def func_skip_tokens(tree):
        return func(tree) if hasattr(tree, 'children') else tree
    if recurse_tokens:
        return list(map(func, tree.children)) if hasattr(tree, 'children') else tree
    return list(map(func_skip_tokens, tree.children))


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
    reachable_set = {non_terminal}
    visit_stack = [non_terminal]
    while len(visit_stack) > 0:
        non_terminal = visit_stack.pop()
        for sequence in rules_dict[non_terminal]:
            for symbol in sequence:
                if isinstance(symbol, NonTerminal) and symbol not in reachable_set:
                    reachable_set.add(symbol)
                    visit_stack.append(symbol)
    return reachable_set


class BiDirectionalList:
    def __init__(self, values=None):
        self._value_to_index = {}
        self._index_to_value = []
        if values:
            for value in values:
                self.append(value)

    def append(self, value):
        if value not in self._value_to_index:
            self._value_to_index[value] = len(self._index_to_value)
            self._index_to_value.append(value)

    def index(self, value):
        return self._value_to_index[value]

    def __getitem__(self, index):
        return self._index_to_value[index]

    def __len__(self):
        return len(self._value_to_index)

    def clear(self):
        self._value_to_index = {}
        self._index_to_value = []

    def __repr__(self):
        return repr(self._index_to_value)


class CustomBNFParser:
    _grammar_file_path = os.path.join('data', 'bnf_lang', 'bnf.lark')

    def __init__(self, grammar_file_path=None):
        self._grammar_file_path = CustomBNFParser._grammar_file_path if grammar_file_path is None else grammar_file_path
        with open(self._grammar_file_path) as f:
            self.parser = lark.Lark(f)

    def parse_file(self, text_file_path, start=None):
        with open(text_file_path) as f:
            return self.parse(f.read(), start=start)

    def parse(self, text, start=None):
        tree = self.parser.parse(text, start=start)
        rules_dict = {}
        syntax = tree.children[0]
        for syntax_child_index in range(1, len(syntax.children), 3):
            syntax_child = syntax.children[syntax_child_index]
            if syntax_child.data == 'comment':
                continue
            assert syntax_child.data == 'rule', 'syntax_child.data: %s' % syntax_child.data
            rule = syntax_child
            non_terminal, expression = rule.children[0], rule.children[4]
            rule_text = non_terminal.children[1]
            rule_name_obj = NonTerminal(''.join([token.value for token in rule_text.children]))

            expression_obj = BiDirectionalList()
            for sequence_index in range(0, len(expression.children), 4):
                sequence = expression.children[sequence_index]
                sequence_obj = []
                for term_index in range(0, len(sequence.children), 2):
                    term = sequence.children[term_index]
                    if term.children[0].data == 'terminal':
                        terminal = term.children[0]
                        text_i = terminal.children[1]
                        term_obj = Terminal(
                            decode_escapes(''.join([token.value for token in text_i.children])))
                    else:  # term.children[0].data == 'non_terminal'
                        rhs_non_terminal = term.children[0]
                        rhs_rule_text = rhs_non_terminal.children[1]
                        term_obj = NonTerminal(''.join([token.value for token in rhs_rule_text.children]))
                    sequence_obj.append(term_obj)
                expression_obj.append(tuple(sequence_obj))
            rules_dict[rule_name_obj] = expression_obj

        return tree, rules_dict


class SimpleTreeActionGetter:
    def __init__(self, rules_dict):
        self.rules_dict = rules_dict
        self.actions = []

    def collect_actions(self, tree):
        non_terminal = NonTerminal(tree.data)
        action = self.rules_dict[non_terminal].index(tree.used_rule)
        self.actions.append((non_terminal, action))
        _ = recurse_tree(tree, self.collect_actions)

    def reset(self):
        self.actions = []
