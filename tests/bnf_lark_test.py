import os

import tree_gan

text_file_path = os.path.join('..', 'data', 'reduced_C_lang', 'reduced_C_lang.bnf')
my_bnf_parser = tree_gan.CustomBNFParser()
tree, rules_dict = my_bnf_parser.parse_file(text_file_path)

print(tree_gan.SimpleTree.pretty(tree))
print('---------------------------------')
tree_gan.CustomBNFParser.print_rules_dict(rules_dict)
print('---------------------------------')

for expression in rules_dict.values():
    for sequence in expression:
        for symbol in sequence:
            if isinstance(symbol, tree_gan.NonTerminal):
                assert symbol in rules_dict, '%s is not in dict!' % str(symbol)


def visit(non_terminal, rules_dict):
    # rules_dict: {NonTerminal : expression}
    reachable_set = {non_terminal}
    visit_stack = [non_terminal]
    while len(visit_stack) > 0:
        non_terminal = visit_stack.pop()
        for sequence in rules_dict[non_terminal]:
            for symbol in sequence:
                if isinstance(symbol, tree_gan.NonTerminal) and symbol not in reachable_set:
                    reachable_set.add(symbol)
                    visit_stack.append(symbol)
    return reachable_set


unreachable_set = set(rules_dict.keys()) - visit(tree_gan.NonTerminal('start'), rules_dict)
print('\n'.join([str(non_terminal) for non_terminal in unreachable_set]))
