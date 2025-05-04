from typing import List, Tuple
from transformers.models.bert import BertForMaskedLM, BertTokenizer
from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizer
import torch
import difflib
from copy import deepcopy
import torch.nn as nn


def get_span(seq1, seq2):
    """
    This implementation comes from official CrowS repository.
    This function extract spans that are shared between two sequences.
    """
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    for op in matcher.get_opcodes():
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]
    return template1, template2


class BaseMaskedLM(nn.Module):
    def __init__(self, model_name: str,device) -> None:
        super().__init__()

    def dual_tokenize(self, sentences1: List[str], sentences2: List[str]):
        full_sent1_masked,full_sent2_masked=[],[]
        full_sent1_attn_mask,full_sent2_attn_mask=[],[]
        full_sent1_mask_index,full_sent2_mask_index=[],[]
        full_sent1_mask_id,full_sent2_mask_id=[],[]
        for sent1,sent2 in zip(sentences1,sentences2):
            all_sent1_masked,all_sent2_masked=[],[]
            all_sent1_mask_index,all_sent2_mask_index=[],[]
            all_sent1_mask_id,all_sent2_mask_id=[],[]
            tokenized=self.tokenizer([sent1,sent2],padding=True)
            sent1_ids,sent1_mask=tokenized["input_ids"][0],tokenized["attention_mask"][0]
            sent2_ids,sent2_mask=tokenized["input_ids"][1],tokenized["attention_mask"][1]
            template1,template2=get_span(sent1_ids,sent2_ids)
            for pos1,pos2 in zip(template1,template2):
                if sent1_ids[pos1] in self.other_token_ids:
                    continue
                sent1_copy=deepcopy(sent1_ids)
                sent2_copy=deepcopy(sent2_ids)
                all_sent1_mask_index.append(pos1)
                all_sent2_mask_index.append(pos2)
                sent1_copy[pos1]=self.mask_token_id
                sent2_copy[pos2]=self.mask_token_id
                all_sent1_masked.append(sent1_copy)
                all_sent2_masked.append(sent2_copy)
                all_sent1_mask_id.append(sent1_ids[pos1])
                all_sent2_mask_id.append(sent1_ids[pos2])
            all_sent1_masked=torch.LongTensor(all_sent1_masked)  # [N_MASK x L]
            sent1_mask=torch.LongTensor(sent1_mask).unsqueeze(0).repeat(all_sent1_masked.shape[0],1)
            all_sent2_masked=torch.LongTensor(all_sent2_masked)  # [N_MASK x L]
            sent2_mask=torch.LongTensor(sent2_mask).unsqueeze(0).repeat(all_sent2_masked.shape[0],1)
            all_sent1_mask_index=torch.LongTensor(all_sent1_mask_index)
            all_sent2_mask_index=torch.LongTensor(all_sent2_mask_index)
            all_sent1_mask_id=torch.LongTensor(all_sent1_mask_id)
            all_sent2_mask_id=torch.LongTensor(all_sent2_mask_id)
            full_sent1_masked.append(all_sent1_masked.to(self.device))
            full_sent2_masked.append(all_sent2_masked.to(self.device))
            full_sent1_attn_mask.append(sent1_mask.to(self.device))
            full_sent2_attn_mask.append(sent2_mask.to(self.device))
            full_sent1_mask_index.append(all_sent1_mask_index.to(self.device))
            full_sent2_mask_index.append(all_sent2_mask_index.to(self.device))
            full_sent1_mask_id.append(all_sent1_mask_id.to(self.device))
            full_sent2_mask_id.append(all_sent2_mask_id.to(self.device))
        return (
            full_sent1_masked,
            full_sent1_attn_mask,
            full_sent1_mask_index,
            full_sent1_mask_id,
            full_sent2_masked,
            full_sent2_attn_mask,
            full_sent2_mask_index,
            full_sent2_mask_id,
        )


class BertModel(BaseMaskedLM):
    def __init__(self, model_name: str = "bert-base-uncased",device="cpu") -> None:
        super().__init__(model_name,device)
        self.device=device
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            model_name)
        self.model: BertForMaskedLM = BertForMaskedLM.from_pretrained(
            model_name).to(device)
        self.other_token_ids = self.tokenizer.convert_tokens_to_ids([
            self.tokenizer.unk_token,
            self.tokenizer.sep_token,
            self.tokenizer.pad_token,
            self.tokenizer.cls_token,
        ])
        self.mask_token_id = self.tokenizer._convert_token_to_id(self.tokenizer.mask_token)
    
    def forward(self,input_ids,attention_mask):
        return self.model(input_ids=input_ids,attention_mask=attention_mask).logits

class RobertaModel(BaseMaskedLM):
    def __init__(self, model_name: str = "roberta-base",device="cpu") -> None:
        super().__init__(model_name,device)
        self.device=device
        self.tokenizer: RobertaTokenizer = RobertaTokenizer.from_pretrained(
            model_name)
        self.model: RobertaForMaskedLM = RobertaForMaskedLM.from_pretrained(
            model_name).to(device)
        self.other_token_ids = self.tokenizer.convert_tokens_to_ids([
            self.tokenizer.unk_token,
            self.tokenizer.sep_token,
            self.tokenizer.pad_token,
            self.tokenizer.cls_token,
        ])
        self.mask_token_id = self.tokenizer._convert_token_to_id(self.tokenizer.mask_token)
    
    def forward(self,input_ids,attention_mask):
        return self.model(input_ids=input_ids,attention_mask=attention_mask).logits


if __name__=="__main__":
    device="cuda"
    model=BertModel(device=device)
    sent1_ids,sent1_mask,sent1_mask_index,sent1_mask_id,sent2_ids,sent2_mask,sent2_mask_index,sent2_mask_id=model.dual_tokenize(["I like flower"],["I don't hate flower"])
    print(sent1_ids[0])
    print(sent1_mask[0])
    print(sent1_mask_index[0])
    print(sent2_ids[0])
    print(sent2_mask[0])
    print(sent2_mask_index[0])
    logits=model(sent1_ids[0],sent1_mask[0])
    print(logits.shape)


