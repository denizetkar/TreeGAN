from collections import namedtuple
import os

from lark import Lark

import utils

grammar_file_path = os.path.join('..', 'data', 'lark_files', 'bnf.lark')
with open(grammar_file_path) as f:
    parser = Lark(f)

text_file_path = os.path.join('..', 'data', 'bnf_files', 'reduced_C_lang.bnf')
with open(text_file_path) as f:
    tree = parser.parse(f.read())

print(utils.pretty(tree))
print('---------------------------------')

rules_dict = {}
syntax = tree.children[0]
for syntax_child in syntax.children:
    if syntax_child.data == 'comment':
        continue
    rule = syntax_child
    non_terminal, expression = rule.children[0], rule.children[4]
    rule_text = non_terminal.children[1]
    rule_name_obj = utils.NonTerminal(''.join([token.value for token in rule_text.children]))

    expression_obj = []
    for sequence_index in range(0, len(expression.children), 4):
        sequence = expression.children[sequence_index]
        sequence_obj = []
        for term_index in range(0, len(sequence.children), 2):
            term = sequence.children[term_index]
            if term.children[0].data == 'terminal':
                terminal = term.children[0]
                text_i = terminal.children[1]
                term_obj = utils.Terminal(utils.decode_escapes(''.join([token.value for token in text_i.children])))
            else:  # term.children[0].data == 'non_terminal'
                rhs_non_terminal = term.children[0]
                rhs_rule_text = rhs_non_terminal.children[1]
                term_obj = utils.NonTerminal(''.join([token.value for token in rhs_rule_text.children]))
            sequence_obj.append(term_obj)
        expression_obj.append(sequence_obj)

    rules_dict[rule_name_obj] = expression_obj
utils.print_rules_dict(rules_dict)
print('---------------------------------')

for expression in rules_dict.items():
    for sequence in expression:
        for symbol in sequence:
            if isinstance(symbol, utils.NonTerminal):
                assert symbol in rules_dict, '%s is not in dict!' % str(symbol)
unreachable_set = set(rules_dict.keys()) - utils.visit(utils.NonTerminal('start'), rules_dict)
print('\n'.join([str(non_terminal) for non_terminal in unreachable_set]))
