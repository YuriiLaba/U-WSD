import torch


def get_hidden_states(model_output):
    hidden_states = model_output["hidden_states"]

    hidden_states = torch.stack(hidden_states, dim=0)
    hidden_states = torch.squeeze(hidden_states, dim=1)
    hidden_states = hidden_states.permute(1, 0, 2)

    return hidden_states


def get_last_hidden_state(hidden_states):
    return hidden_states[:, -1]