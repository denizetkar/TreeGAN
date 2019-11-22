import os
import time

import numpy as np
from scipy.optimize import nnls
import torch
import torch.nn as nn

from tree_gan import ActionSequenceDataset, TreeGenerator, NonTerminal, Enumerator

START = time.process_time()
reduced_C_dir = os.path.join('data', 'reduced_C_lang')

reduced_C_bnf_path = os.path.join(reduced_C_dir, 'reduced_C_lang.bnf')
reduced_C_lark_path = os.path.join(reduced_C_dir, 'reduced_C_lang.lark')
reduced_C_text_dir = os.path.join(reduced_C_dir, 'text_files')
reduced_C_action_getter_path = os.path.join(reduced_C_dir, 'action_getter.pickle')
reduced_C_action_sequences_dir = os.path.join(reduced_C_dir, 'action_sequence_files')

a_s_dataset = ActionSequenceDataset(reduced_C_bnf_path, reduced_C_lark_path, reduced_C_text_dir,
                                    reduced_C_action_getter_path, reduced_C_action_sequences_dir)
action_sequence = a_s_dataset[a_s_dataset.index('code.c')]
print('ELAPSED TIME (sec): ' + str(time.process_time() - START))
print('---------------------------------')
print(a_s_dataset.action_getter.construct_text(action_sequence))
print('---------------------------------')
tree_generator = TreeGenerator(a_s_dataset.action_getter.rules_dict, a_s_dataset.action_getter.symbol_names,
                               a_s_dataset.action_getter.action_offsets)

with torch.no_grad():
    nn.init.zeros_(tree_generator.action_layer.bias)
    non_terminal_ids = Enumerator()
    bias_increment = 0.8
    # Discourage producing 'preprocessing_file' symbol
    preprocessing_file_bias = -6
    preprocessing_file_id = tree_generator.symbol_names.index(NonTerminal('preprocessing_file'))
    for non_terminal_id, rules in tree_generator.rules_dict.items():
        non_terminal_ids.append(non_terminal_id)
        for i, rule in enumerate(rules):
            if preprocessing_file_id in rule:
                action = tree_generator.action_offsets[non_terminal_id] + i
                tree_generator.action_layer.bias[action] += preprocessing_file_bias
    # Find the expected length of each rule assuming that each rule is equally likely to be picked
    A = np.eye(len(non_terminal_ids))
    b = np.zeros(len(non_terminal_ids))
    for non_terminal_id, rules in tree_generator.rules_dict.items():
        rule_non_terminal_ids_count = {}
        rule_terminals_length = 0
        for rule in rules:
            for symbol_id in rule:
                symbol = tree_generator.symbol_names[symbol_id]
                if isinstance(symbol, NonTerminal):
                    matrix_id = non_terminal_ids.index(symbol_id)
                    rule_non_terminal_ids_count.setdefault(matrix_id, 0)
                    rule_non_terminal_ids_count[matrix_id] += 1
                else:
                    rule_terminals_length += len(symbol.name)
        A[non_terminal_ids.index(non_terminal_id), list(rule_non_terminal_ids_count.keys())] -= np.array(
            list(rule_non_terminal_ids_count.values())) / len(rules)
        b[non_terminal_ids.index(non_terminal_id)] += rule_terminals_length / len(rules)
    x, _ = nnls(A, b, maxiter=A.shape[0] ** 2)
    # Increase bias for every rule inversely proportional to their expected lengths
    for non_terminal_id, rules in tree_generator.rules_dict.items():
        rule_lengths = []
        for rule in rules:
            rule_length = sum(x[non_terminal_ids.index(symbol_id)]
                              if isinstance(tree_generator.symbol_names[symbol_id], NonTerminal)
                              else torch.tensor(len(tree_generator.symbol_names[symbol_id].name), dtype=torch.float)
                              for symbol_id in rule)
            rule_lengths.append(rule_length)
        rule_lengths = np.stack(rule_lengths)
        rule_biases = np.exp(rule_lengths.min() - rule_lengths)
        for i, rule_bias in enumerate(rule_biases):
            action = tree_generator.action_offsets[non_terminal_id] + i
            tree_generator.action_layer.bias[action] += bias_increment * rule_bias
    # Generate an action sequence (equivalent to parse tree or text file)
    generated_action_sequence, _ = tree_generator()

print(a_s_dataset.action_getter.construct_text(generated_action_sequence.squeeze().numpy().tolist()))
print('---------------------------------')
pass
