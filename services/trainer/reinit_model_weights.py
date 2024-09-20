from transformers import AutoModel
import torch.nn as nn


class ModelWithRandomizingSomeWeights(nn.Module):
    def __init__(self, model, reinit_n_layers=0):
        super().__init__()
        self.model = model
        self.reinit_n_layers = reinit_n_layers
        if reinit_n_layers > 0:
            self._do_reinit()

    def _do_reinit(self):
        self.model.pooler.dense.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
        self.model.pooler.dense.bias.data.zero_()
        for param in self.model.pooler.parameters():
            param.requires_grad = True

        for n in range(self.reinit_n_layers):
            self.model.encoder.layer[-(n+1)].apply(self._init_weight_and_bias)

    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask):
        raw_output = self.model(input_ids, attention_mask)
        return raw_output
