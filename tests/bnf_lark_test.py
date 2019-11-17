import os

import utils

text_file_path = os.path.join('..', 'data', 'reduced_C_lang', 'reduced_C_lang.bnf')
my_bnf_parser = utils.CustomBNFParser()
tree, rules_dict = my_bnf_parser.parse_file(text_file_path)

print(utils.pretty(tree))
print('---------------------------------')
utils.print_rules_dict(rules_dict)
print('---------------------------------')

for expression in rules_dict.values():
    for sequence in expression:
        for symbol in sequence:
            if isinstance(symbol, utils.NonTerminal):
                assert symbol in rules_dict, '%s is not in dict!' % str(symbol)
unreachable_set = set(rules_dict.keys()) - utils.visit(utils.NonTerminal('start'), rules_dict)
print('\n'.join([str(non_terminal) for non_terminal in unreachable_set]))
