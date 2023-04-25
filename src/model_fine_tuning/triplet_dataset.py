import torch


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, anchor, positive, negative, tokenizer, seq_len=128):
        self.anchor = anchor
        self.positive = positive
        self.negative = negative
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, idx):
        anchor = str(self.anchor[idx])
        positive = str(self.positive[idx])
        negative = str(self.negative[idx])

        tokenized_anchor = self.tokenizer(
            anchor,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        )
        tokenized_positive = self.tokenizer(
            positive,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        )
        tokenized_negative = self.tokenizer(
            negative,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        )

        return {"anchor_ids": torch.tensor(tokenized_anchor["input_ids"], dtype=torch.long),
                "anchor_mask": torch.tensor(tokenized_anchor["attention_mask"], dtype=torch.long),

                "positive_ids": torch.tensor(tokenized_positive["attention_mask"], dtype=torch.long),
                "positive_mask": torch.tensor(tokenized_positive["attention_mask"], dtype=torch.long),

                "negative_ids": torch.tensor(tokenized_negative["attention_mask"], dtype=torch.long),
                "negative_mask": torch.tensor(tokenized_negative["attention_mask"], dtype=torch.long),
                }
