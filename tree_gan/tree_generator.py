from collections import namedtuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from tree_gan.utils import NonTerminal, Terminal

DerivationSymbol = namedtuple('DerivationSymbol', ['name', 'parent_action'])


class TreeGenerator(nn.Module):
    def __init__(self, rules_dict, symbol_names, action_offsets, rand_initial_state_func=None, start='start',
                 action_embedding_size=None, hidden_size=None, batch_first=False, rnn_cls=None, rnn_kwargs=None):
        super(TreeGenerator, self).__init__()
        self.rules_dict = rules_dict
        self.symbol_names = symbol_names
        self.action_offsets = action_offsets
        self.rand_initial_state_func = rand_initial_state_func
        self.start_id = symbol_names.index(NonTerminal(start))
        num_of_rules = sum(len(rules) for rules in rules_dict.values())
        if action_embedding_size is None:
            action_embedding_size = (num_of_rules - 1) // 4 + 1
        if hidden_size is None:
            hidden_size = action_embedding_size
        self.batch_first = batch_first
        if rnn_cls is None:
            rnn_cls = nn.GRU
            if rnn_kwargs is None:
                rnn_kwargs = dict(num_layers=2, dropout=0.1)
            if self.rand_initial_state_func is None:
                num_layers = rnn_kwargs.get('num_layers', 1)
                num_directions = int(rnn_kwargs.get('bidirectional', False)) + 1

                def func():
                    return torch.randn((num_layers * num_directions, 1, hidden_size)) / (
                                num_layers * num_directions * 1 * hidden_size) ** 0.5
                self.rand_initial_state_func = func
        else:
            if rnn_kwargs is None:
                rnn_kwargs = dict()
            assert self.rand_initial_state_func is not None, 'rand_initial_state_func is not known!'
        rnn_kwargs['batch_first'] = batch_first
        input_size = action_embedding_size * 2
        self.rnn = rnn_cls(input_size, hidden_size, **rnn_kwargs)

        self.padding_action = -1
        self.action_offset = 1

        self.action_embeddings = nn.Embedding(num_of_rules + self.action_offset, action_embedding_size,
                                              padding_idx=self.padding_action + self.action_offset)
        self.action_layer = nn.Linear(hidden_size, num_of_rules)
        # nn.ParameterDict DOES NOT ACCEPT NON-STRING's AS KEY !!!!!!!!!!!!!!!!!!
        self.action_masks = nn.ParameterDict()
        for non_terminal_id, rules in self.rules_dict.items():
            # Create the action mask for each non-terminal symbol
            action_mask = nn.Parameter(torch.zeros(num_of_rules, dtype=torch.bool), requires_grad=False)
            first_action = self.action_offsets[non_terminal_id]
            last_action = first_action + len(rules)
            action_mask[first_action:last_action] = 1
            self.action_masks[str(non_terminal_id)] = action_mask
        self.device = None

    def to(self, *args, **kwargs):
        if args and (isinstance(args[0], torch.device) or ('cuda' in args[0]) or ('cpu' in args[0])):
            self.device = args[0]
        elif kwargs and 'device' in kwargs:
            self.device = kwargs['device']

        return super(TreeGenerator, self).to(*args, **kwargs)

    def forward(self):
        # TODO: take batch_size as input and vectorize generation process !!!!
        # output: [seq_len, 1] OR [1, seq_len]
        seq_len_dim_index = int(self.batch_first)
        prev_state = self.rand_initial_state_func().to(self.device)
        prev_action = torch.tensor([self.padding_action], device=self.device)
        symbol_stack = [DerivationSymbol(self.start_id, prev_action)]

        output = []
        while symbol_stack:
            symbol_id, parent_action = symbol_stack.pop()
            if isinstance(self.symbol_names[symbol_id], Terminal):
                continue

            prev_action_embedding = self.action_embeddings(prev_action + self.action_offset)
            parent_action_embedding = self.action_embeddings(parent_action + self.action_offset)
            current_input = torch.cat([prev_action_embedding, parent_action_embedding], dim=-1).unsqueeze(
                seq_len_dim_index)
            out, prev_state = self.rnn(current_input, prev_state)
            out = out.squeeze(seq_len_dim_index)
            action_log_prob = torch.log_softmax(self.action_layer(out), dim=-1)
            action_dist = Categorical(logits=action_log_prob.masked_fill_(
                self.action_masks[str(symbol_id)].logical_not().unsqueeze(0), float('-inf')))
            action = action_dist.sample()
            output.append(action.unsqueeze(seq_len_dim_index))

            for child_id in reversed(self.rules_dict[symbol_id][action.item() - self.action_offsets[symbol_id]]):
                if isinstance(self.symbol_names[symbol_id], NonTerminal):
                    symbol_stack.append(DerivationSymbol(child_id, parent_action=action))

            prev_action = action

        return torch.cat(output, dim=seq_len_dim_index), prev_state
