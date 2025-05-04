from dataset import CrowSDataset
from models import BertModel
import torch


DEVICE="cuda"


def main():
    model=BertModel(device=DEVICE).eval()
    dataset=CrowSDataset()
    log_softmax=torch.nn.LogSoftmax(dim=-1)
    for x in dataset:
        sent1_ids,sent1_mask,sent1_mask_index,sent1_mask_id,sent2_ids,sent2_mask,sent2_mask_index,sent2_mask_id=model.dual_tokenize([x[0]],[x[1]])
        N=sent1_ids[0].shape[0]
        with torch.no_grad():
            sent1_logits=model(sent1_ids[0],sent1_mask[0])  # [N x L x dict_size]
            sent2_logits=model(sent2_ids[0],sent2_mask[0])  # [N x L x dict_size]
        seq_index=torch.linspace(0,N-1,N,dtype=torch.long).to(DEVICE)  # [L]
        sent1_mask_logits=sent1_logits[seq_index,sent1_mask_index[0]]  # [N x dict_size]
        sent2_mask_logits=sent2_logits[seq_index,sent2_mask_index[0]]  # [N x dict_size]
        sent1_log_softmax_logits=log_softmax(sent1_mask_logits)  # [N x dict_size]
        sent2_log_softmax_logits=log_softmax(sent2_mask_logits)  # [N x dict_size]
        sent1_score=sent1_log_softmax_logits[seq_index,sent1_mask_id[0]].sum()  # scalar
        sent2_score=sent2_log_softmax_logits[seq_index,sent2_mask_id[0]].sum()  # scalar
        print(sent1_score)
        print(sent2_score)




if __name__=="__main__":
    main()