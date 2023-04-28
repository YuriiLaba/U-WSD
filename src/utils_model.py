import torch


def _get_hidden_states(model_output):
    hidden_states = model_output["hidden_states"]

    hidden_states = torch.stack(hidden_states, dim=0)
    hidden_states = torch.squeeze(hidden_states, dim=1)
    hidden_states = hidden_states.permute(1, 0, 2)

    return hidden_states


def get_last_hidden_state(hidden_states):
    return hidden_states[:, -1]


def _get_model_output(model, tokenized_input_text):
    with torch.no_grad():
        model_output = model(**tokenized_input_text)
    return model_output


def _tokenize_text(tokenizer, input_text, device):
    return tokenizer(input_text, return_tensors='pt').to(device)


def run_inference(model, tokenizer, text, device):
    tokenized_text = _tokenize_text(tokenizer, text, device)
    model_output = _get_model_output(model, tokenized_text)
    return tokenized_text, _get_hidden_states(model_output)
