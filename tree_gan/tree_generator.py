import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from tree_gan import learning_utils
from tree_gan import parse_utils


class TreeGenerator(nn.Module):
    """
    Effectively the batch size for both 'evaluate' and 'generate' functions are always 1
    AND never use bidirectional RNN !!!
    """
    def __init__(self, action_getter, rand_initial_state_func=None, start='start', action_embedding_size=None,
                 hidden_size=None, batch_first=False, rnn_cls=None, rnn_kwargs=None):
        super().__init__()
        self.rules_dict = action_getter.rules_dict
        self.symbol_names = action_getter.symbol_names
        self.action_offsets = action_getter.action_offsets
        self.non_terminal_ids = action_getter.non_terminal_ids
        self.rand_initial_state_func = rand_initial_state_func
        self.start_id = self.symbol_names.index(parse_utils.NonTerminal(start))
        num_of_rules = sum(len(rules) for rules in self.rules_dict.values())
        if action_embedding_size is None:
            action_embedding_size = (num_of_rules - 1) // 4 + 1
        if hidden_size is None:
            hidden_size = action_embedding_size
        self.batch_first = batch_first
        if rnn_cls is None:
            rnn_cls = nn.GRU
            if rnn_kwargs is None:
                rnn_kwargs = dict(num_layers=2, dropout=0.0)
            if self.rand_initial_state_func is None:
                num_layers = rnn_kwargs.get('num_layers', 1)

                def func():
                    return torch.randn((num_layers, 1, hidden_size)) / (
                                num_layers * 1 * hidden_size) ** 0.5
                self.rand_initial_state_func = func
        else:
            if rnn_kwargs is None:
                rnn_kwargs = dict()
            assert self.rand_initial_state_func is not None, 'rand_initial_state_func is not known!'
        rnn_kwargs['batch_first'] = batch_first
        input_size = action_embedding_size * 2
        self.rnn = rnn_cls(input_size, hidden_size, **rnn_kwargs)

        self.action_embeddings = nn.Embedding(num_of_rules + learning_utils.UNIVERSAL_ACTION_OFFSET,
                                              action_embedding_size,
                                              padding_idx=learning_utils.PADDING_ACTION +
                                                          learning_utils.UNIVERSAL_ACTION_OFFSET)
        self.action_layer = nn.Linear(hidden_size, num_of_rules)
        self.value_layer = nn.Linear(hidden_size, 1)
        self.action_masks = nn.Parameter(torch.empty(len(self.non_terminal_ids), num_of_rules, dtype=torch.bool),
                                         requires_grad=False)
        for non_terminal_id, rules in self.rules_dict.items():
            # Create the action mask for each non-terminal symbol
            action_mask = torch.zeros(num_of_rules, dtype=torch.bool)
            first_action = self.action_offsets[non_terminal_id]
            last_action = first_action + len(rules)
            action_mask[first_action:last_action] = 1
            self.action_masks[self.non_terminal_ids.index(non_terminal_id)].copy_(action_mask.data, non_blocking=True)
        self.lt_rewards_norm_layer = nn.BatchNorm1d(1, affine=False)
        self.advantages_norm_layer = nn.BatchNorm1d(1, affine=False)
        self.device = None

    def cpu(self):
        res = super().cpu()
        self.device = 'cpu'
        return res

    def cuda(self, device=None):
        res = super().cuda(device=device)
        self.device = 'cuda' if device is None else device
        return res

    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)

        if args and (isinstance(args[0], torch.device) or ('cuda' in args[0]) or ('cpu' in args[0])):
            self.device = args[0]
        elif kwargs and 'device' in kwargs:
            self.device = kwargs['device']

        return res

    def forward(self, max_sequence_length=None):
        # output dimensions: (seq_len, [specific_size])
        max_sequence_length = float('inf') if max_sequence_length is None else max_sequence_length
        seq_len_dim_index = int(self.batch_first)
        initial_state = self.rand_initial_state_func().to(self.device, non_blocking=True)
        prev_state = initial_state
        prev_action = torch.tensor([learning_utils.PADDING_ACTION], device=self.device)
        symbol_stack = [parse_utils.DerivationSymbol(self.start_id, parent_action=prev_action)]

        action_list, parent_action_list, log_prob_list, value_list = [], [], [], []
        while True:
            if not symbol_stack:
                # Action generation ended before 'max_sequence_length'
                next_value = torch.zeros_like(value)
                value_list.append(next_value.unsqueeze(seq_len_dim_index))
                break
            if len(action_list) > max_sequence_length:
                # Action generation did not end before 'max_sequence_length'
                action_list.pop()
                parent_action_list.pop()
                log_prob_list.pop()
                break

            symbol_id, parent_action = symbol_stack.pop()

            prev_action_embedding = self.action_embeddings(prev_action + learning_utils.UNIVERSAL_ACTION_OFFSET)
            parent_action_embedding = self.action_embeddings(parent_action + learning_utils.UNIVERSAL_ACTION_OFFSET)
            current_input = torch.cat([prev_action_embedding, parent_action_embedding], dim=-1).unsqueeze(
                seq_len_dim_index)
            # Get next action
            out, prev_state = self.rnn(current_input, prev_state)
            out = out.squeeze(seq_len_dim_index)
            log_prob = torch.log_softmax(self.action_layer(out), dim=-1)
            action_dist = Categorical(logits=log_prob.masked_fill(
                self.action_masks[self.non_terminal_ids.index(symbol_id)].bitwise_not().unsqueeze(0), float('-inf')))
            action = action_dist.sample()
            # Get next (state) value
            value = self.value_layer(out).squeeze(-1)
            action_list.append(action.unsqueeze(seq_len_dim_index))
            parent_action_list.append(parent_action.unsqueeze(seq_len_dim_index))
            log_prob_list.append(action_dist.logits.unsqueeze(seq_len_dim_index))
            value_list.append(value.unsqueeze(seq_len_dim_index))

            for child_id in reversed(self.rules_dict[symbol_id][action.item() - self.action_offsets[symbol_id]]):
                if isinstance(self.symbol_names[child_id], parse_utils.NonTerminal):
                    symbol_stack.append(parse_utils.DerivationSymbol(child_id, parent_action=action))

            prev_action = action

        batch_dim_index = 1 - seq_len_dim_index
        actions = torch.cat(action_list, dim=seq_len_dim_index).squeeze(batch_dim_index)
        parent_actions = torch.cat(parent_action_list, dim=seq_len_dim_index).squeeze(batch_dim_index)
        log_probs = torch.cat(log_prob_list, dim=seq_len_dim_index).squeeze(batch_dim_index)
        values = torch.cat(value_list, dim=seq_len_dim_index).squeeze(batch_dim_index)
        return initial_state, actions, parent_actions, log_probs, values

    def evaluate(self, old_initial_gen_state, old_actions, old_parent_actions):
        # input/output dimensions: (seq_len, [specific_size])
        seq_len_dim_index = int(self.batch_first)
        batch_dim_index = 1 - seq_len_dim_index
        old_actions = old_actions.unsqueeze(batch_dim_index)
        old_parent_actions = old_parent_actions.unsqueeze(batch_dim_index)

        prev_action = torch.tensor([learning_utils.PADDING_ACTION], device=self.device).unsqueeze(dim=seq_len_dim_index)
        seq_len = old_actions.shape[seq_len_dim_index]
        old_actions = torch.cat([prev_action, old_actions.narrow(seq_len_dim_index, 0, seq_len - 1)],
                                dim=seq_len_dim_index)

        action_embeddings = self.action_embeddings(old_actions + learning_utils.UNIVERSAL_ACTION_OFFSET)
        parent_action_embeddings = self.action_embeddings(old_parent_actions + learning_utils.UNIVERSAL_ACTION_OFFSET)

        current_input = torch.cat([action_embeddings, parent_action_embeddings], dim=-1)
        out, _ = self.rnn(current_input, old_initial_gen_state)

        log_probs = torch.log_softmax(self.action_layer(out), dim=-1)
        values = self.value_layer(out).squeeze(-1)
        # TODO: mask out irrelevant log_probs with float('-inf') and run it through Categorical(logits=log_probs).logits
        return log_probs.squeeze(batch_dim_index), values.squeeze(batch_dim_index)

    def ppo_losses(self, old_initial_gen_state_list, old_actions_list, old_parent_actions_list,
                   old_log_probs_list, old_values_list, old_lt_rewards_list, eps_clip):
        num_episodes = len(old_actions_list)
        policy_losses, entropy_losses, value_losses = [], [], []
        for episode_index in range(num_episodes):
            old_initial_gen_state = old_initial_gen_state_list[episode_index]
            old_actions = old_actions_list[episode_index]
            old_parent_actions = old_parent_actions_list[episode_index]
            old_log_probs = old_log_probs_list[episode_index]
            old_values = old_values_list[episode_index]
            old_lt_rewards = old_lt_rewards_list[episode_index]
            old_action_log_probs = old_log_probs.gather(dim=-1, index=old_actions.unsqueeze(-1)).squeeze(-1)
            try:
                old_lt_rewards = self.lt_rewards_norm_layer(old_lt_rewards.unsqueeze(-1)).squeeze(-1)
                old_advantages = self.advantages_norm_layer((old_lt_rewards - old_values).unsqueeze(-1)).squeeze(-1)
            except ValueError:
                # Error is most likely due to giving 1 sample to batchnorm layer
                continue
            log_probs, values = self.evaluate(old_initial_gen_state, old_actions, old_parent_actions)

            restriction_mask = torch.isinf(old_log_probs)
            action_dist = Categorical(logits=log_probs.masked_fill(restriction_mask, float('-inf')))
            probs, log_probs = action_dist.probs, action_dist.logits
            # Get new log probabilities of actions from 'log_probs'
            action_log_probs = log_probs.gather(dim=-1, index=old_actions.unsqueeze(-1)).squeeze(-1)
            # Calculate entropy of log probabilities for each time step
            p_log_p_s = probs.masked_fill(restriction_mask, 0.0) * log_probs.masked_fill(restriction_mask, 0.0)
            dist_entropy = -p_log_p_s.sum(-1)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(action_log_probs - old_action_log_probs)

            # Finding Surrogate Loss:
            surr1 = ratios * old_advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * old_advantages
            policy_loss = -torch.min(surr1, surr2)
            entropy_loss = -dist_entropy

            # Finding value loss:
            value_clipped = old_values + torch.clamp(values - old_values, -eps_clip, eps_clip)
            v_loss1 = F.smooth_l1_loss(values, old_lt_rewards, reduction='none')
            v_loss2 = F.smooth_l1_loss(value_clipped, old_lt_rewards, reduction='none')
            value_loss = torch.max(v_loss1, v_loss2)

            policy_losses.append(policy_loss)
            entropy_losses.append(entropy_loss)
            value_losses.append(value_loss)

        if len(policy_losses) == 0:
            return None, None, None

        return torch.cat(policy_losses), torch.cat(entropy_losses), torch.cat(value_losses)
