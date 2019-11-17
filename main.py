import os
import pickle

from lark import Lark

import utils

reduced_C_dir = os.path.join('data', 'reduced_C_lang')
# Get rule dictionary of the reduced C language
# First check if parse result already exists, if not then parse the reduced C language grammar
reduced_C_parse_results_path = os.path.join(reduced_C_dir, 'reduced_C_parse_results.pickle')
if os.path.exists(reduced_C_parse_results_path):
    with open(reduced_C_parse_results_path, 'rb') as f:
        reduced_C_parse_results = pickle.load(f)
else:
    my_bnf_parser = utils.CustomBNFParser()
    reduced_C_bnf_path = os.path.join(reduced_C_dir, 'reduced_C_lang.bnf')
    reduced_C_parse_results = my_bnf_parser.parse_file(reduced_C_bnf_path)
    with open(reduced_C_parse_results_path, 'wb') as f:
        pickle.dump(reduced_C_parse_results, f)
reduced_C_tree, reduced_C_rules_dict = reduced_C_parse_results
del reduced_C_tree

# Get parse tree of the source file written in reduced C language
reduced_C_lark_path = os.path.join(reduced_C_dir, 'reduced_C_lang.lark')
with open(reduced_C_lark_path) as f:
    reduced_C_parser = Lark(f, keep_all_tokens=True)
reduced_C_action_getter = utils.SimpleTreeActionGetter(reduced_C_rules_dict)

# TODO: Loop over all source files and extract the action sequence of their parse trees
# First check if action sequence of parse tree of the source (text) file already exists, if not then calculate it
source_filename = 'code.c'
source_path = os.path.join(reduced_C_dir, 'text_files', source_filename)
source_action_sequence_path = os.path.join(reduced_C_dir, 'action_sequence_files', source_filename + '.pickle')
if os.path.exists(source_action_sequence_path):
    with open(source_action_sequence_path, 'rb') as f:
        action_sequence = pickle.load(f)
else:
    with open(source_path) as f:
        # Parse the source (text) file
        source_tree = reduced_C_parser.parse(f.read())
        source_tree = utils.SimpleTree.from_lark_tree(source_tree)
        # Get sequence of actions taken by each non-terminal symbol in 'prefix DFS left-to-right' order
        reduced_C_action_getter.collect_actions(source_tree)
        action_sequence = reduced_C_action_getter.actions
        reduced_C_action_getter.reset()
    with open(source_action_sequence_path, 'wb') as f:
        pickle.dump(action_sequence, f)

pass
