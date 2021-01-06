import random
import funcy as fp
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from transformers import PreTrainedTokenizer, PreTrainedModel, AdamW, get_linear_schedule_with_warmup


from ..pipeline.general import clean_text


class TransformerClassifier(nn.Module):
    """Transformer-Based Model for Classification Tasks.
    """

    def __init__(self, pretrained_model_class: PreTrainedModel, pretrained_model_name: str, extra_layers: List[int],
                 dropout_layers: List[float] = None, freeze: bool = False):
        """
        @param  pretrained_model_class: an object of a pre trained model class (e.g., BertModel)
        @param  pretrained_model_name: a pretrained model path (e.g., 'neuralmind/bert-base-portuguese-cased')
        @param  freeze (bool): whether the model should be fine tuned (True) or not (False).
        """
        super(TransformerClassifier, self).__init__()
        # Instantiate  model
        self.model = pretrained_model_class.from_pretrained(pretrained_model_name)

        dropout_layers = dropout_layers or [0. for _ in extra_layers]
        assert len(extra_layers) == len(dropout_layers), 'Extra Layers and Dropout Layers should have the same length'

        # Adds the size of the output layer
        all_layers = [self.model.config.hidden_size] + extra_layers + [3]
        dropout_layers = [0.] + dropout_layers + [0.]
        # Instantiate layers based on the sizes received
        layers_instances = fp.lflatten([[nn.Linear(prev, layer), nn.ReLU()] +
                                        ([nn.Dropout(dropout_layers[i])] if dropout_layers[i] > 0 else [])
                                        for i, (layer, prev) in enumerate(fp.with_prev(all_layers)) if prev])
        layers_instances = layers_instances[:-1]  # Remove the last ReLU added.
        self.classifier = nn.Sequential(*layers_instances)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def preprocess(data: np.array, tokenizer: PreTrainedTokenizer, max_len: int,
               unify_html_tags: bool, unify_urls: bool, trim_repeating_spaces: bool,
               unify_hashtags: bool, unify_mentions: bool, unify_numbers: bool,
               trim_repeating_letters: bool, lower_case: bool):
    input_ids = []
    attention_masks = []

    partial_clean_text = partial(clean_text, unify_html_tags=unify_html_tags, unify_urls=unify_urls,
                                 trim_repeating_spaces=trim_repeating_spaces, unify_hashtags=unify_hashtags,
                                 unify_mentions=unify_mentions, unify_numbers=unify_numbers,
                                 trim_repeating_letters=trim_repeating_letters, lower_case=lower_case)

    for sentence in data:
        encoded_sentence = tokenizer.encode_plus(
            text=partial_clean_text(sentence),
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True
        )

        input_ids.append(encoded_sentence.get('input_ids'))
        attention_masks.append(encoded_sentence.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def initialize_model(pretrained_model_class: PreTrainedModel, pretrained_model_name: str, extra_layers: List[int],
                     dropout_layers: List[float], training_len: int, epochs: int = 2, custom_scheduler: bool = True,
                     freeze: bool = False, learning_rate=2e-5):
    transformer_classifier = TransformerClassifier(pretrained_model_class, pretrained_model_name, extra_layers,
                                                   dropout_layers, freeze=freeze)
    transformer_classifier.to(get_device())

    optimizer = AdamW(transformer_classifier.parameters(), lr=learning_rate, eps=1e-8)

    # Total number of training steps
    total_steps = training_len * epochs

    if custom_scheduler:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    return transformer_classifier, optimizer, scheduler


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def evaluate(model, eval_dataloader, loss_fn):
    model.eval()

    eval_fmeasure = []
    eval_loss = []

    for batch in eval_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(get_device()) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        loss = loss_fn(logits, b_labels)
        eval_loss.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()

        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        eval_fmeasure.append(accuracy)

    eval_loss = np.mean(eval_loss)
    eval_fmeasure = np.mean(eval_fmeasure)

    return eval_loss, eval_fmeasure


def predict(model, dataloader):
    model.eval()

    all_logits = []

    for batch in dataloader:
        b_input_ids, b_attn_mask = tuple(t.to(get_device()) for t in batch)[:2]

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)

    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs
