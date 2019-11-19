import os

from lark import Lark

import tree_gan

grammar_file_path = os.path.join('..', 'data', 'reduced_C_lang', 'reduced_C_lang.lark')
with open(grammar_file_path) as f:
    parser = Lark(f, keep_all_tokens=True)

text_file_path = os.path.join('..', 'data', 'reduced_C_lang', 'text_files', 'code.c')
with open(text_file_path) as f:
    tree = parser.parse(f.read())

tree = tree_gan.SimpleTree.from_lark_tree(tree)
print(tree.pretty())
