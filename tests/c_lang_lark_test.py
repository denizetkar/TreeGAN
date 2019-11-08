import os

from lark import Lark

import utils


grammar_file_path = os.path.join('..', 'data', 'lark_files', 'reduced_C_lang.lark')
with open(grammar_file_path) as f:
    parser = Lark(f, keep_all_tokens=True)

text_file_path = os.path.join('..', 'data', 'code.c')
with open(text_file_path) as f:
    tree = parser.parse(f.read())

print(utils.pretty(tree))
