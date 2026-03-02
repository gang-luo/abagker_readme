import os

import torch
import transformers
from tqdm import tqdm

import antiberty
from antiberty import AntiBERTy
from antiberty.utils.general import exists

project_path = os.path.dirname(os.path.realpath(antiberty.__file__))
trained_models_dir = os.path.join(project_path, 'trained_models')

CHECKPOINT_PATH = os.path.join(trained_models_dir, 'AntiBERTy_md_smooth')
VOCAB_FILE = os.path.join(trained_models_dir, 'vocab.txt')

LABEL_TO_SPECIES = {
    0: "Camel",
    1: "Human",
    2: "Mouse",
    3: "Rabbit",
    4: "Rat",
    5: "Rhesus"
}
LABEL_TO_CHAIN = {0: "Heavy", 1: "Light"}

SPECIES_TO_LABEL = {v: k for k, v in LABEL_TO_SPECIES.items()}
CHAIN_TO_LABEL = {v: k for k, v in LABEL_TO_CHAIN.items()}


class AntiBERTyRunner():
    def __init__(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = AntiBERTy.from_pretrained(CHECKPOINT_PATH).to(self.device)
        self.model.eval()

        self.tokenizer = transformers.BertTokenizer(vocab_file=VOCAB_FILE,
                                                    do_lower_case=False)
        print("AntiBERTyRunner initialized")

    def embed(self, sequences, hidden_layer=-1, return_attention=False, max_length = 128,add_special_tokens=True):
        """
        Embed a list of sequences.

        Args:
            sequences (list): list of sequences
            hidden_layer (int): which hidden layer to use (0 to 8)
            return_attention (bool): whether to return attention matrices

        Returns:
            list(torch.Tensor): list of embeddings (one tensor per sequence)

        """
        sequences = [list(s) for s in sequences]
        for s in sequences:
            for i, c in enumerate(s):
                if c == "_":
                    s[i] = "[MASK]"

        sequences = [" ".join(s) for s in sequences]
        tokenizer_out = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding='max_length',      # 强制pad到max_length
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens
        )
        tokens = tokenizer_out["input_ids"].to(self.device)
        attention_mask = tokenizer_out["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=return_attention,
            )

        # gather embeddings
        embeddings = outputs.hidden_states
        embeddings = torch.stack(embeddings, dim=1)
        full_embeddings = embeddings[:,hidden_layer,:,:]

        embeddings = list(embeddings.detach())
        for i, a in enumerate(attention_mask):
            embeddings[i] = embeddings[i][:, a == 1]

        if exists(hidden_layer):
            for i in range(len(embeddings)):
                embeddings[i] = embeddings[i][hidden_layer]

        # gather attention matrices
        if return_attention:
            attentions = outputs.attentions
            attentions = torch.stack(attentions, dim=1)
            attentions = list(attentions.detach())

            # # 这个又是根据token make掉了的
            # for i, a in enumerate(attention_mask):
            #     attentions[i] = attentions[i][:, :, a == 1]
            #     attentions[i] = attentions[i][:, :, :, a == 1]

            return tokens, attention_mask, full_embeddings, embeddings, attentions

        return tokens, attention_mask, full_embeddings, embeddings
