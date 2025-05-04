from dataset import CrowSDataset
from models import BertModel, RobertaModel
import torch
from tqdm import tqdm
import pickle


DEVICE = "cuda"
model = BertModel(device=DEVICE)
# model = RobertaModel(device=DEVICE)


def main():
    score1, score2 = [], []
    dataset = CrowSDataset()
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    for x in tqdm(dataset):
        sent1_ids, sent1_mask, sent1_mask_index, sent1_mask_id, sent2_ids, sent2_mask, sent2_mask_index, sent2_mask_id = model.dual_tokenize([
                                                                                                                                             x[0]], [x[1]])
        N = sent1_ids[0].shape[0]
        with torch.no_grad():
            # [N x L x dict_size]
            sent1_logits = model(sent1_ids[0], sent1_mask[0])
            # [N x L x dict_size]
            sent2_logits = model(sent2_ids[0], sent2_mask[0])
        seq_index = torch.linspace(
            0, N-1, N, dtype=torch.long).to(DEVICE)  # [L]
        sent1_mask_logits = sent1_logits[seq_index,
                                         # [N x dict_size]
                                         sent1_mask_index[0]]
        sent2_mask_logits = sent2_logits[seq_index,
                                         # [N x dict_size]
                                         sent2_mask_index[0]]
        sent1_log_softmax_logits = log_softmax(
            sent1_mask_logits)  # [N x dict_size]
        sent2_log_softmax_logits = log_softmax(
            sent2_mask_logits)  # [N x dict_size]
        sent1_scores = sent1_log_softmax_logits[seq_index, sent1_mask_id[0]].cpu(
        ).tolist()
        sent2_scores = sent2_log_softmax_logits[seq_index, sent2_mask_id[0]].cpu(
        ).tolist()
        score1.append(sent1_scores)
        score2.append(sent2_scores)
    n_masked = sum([len(x) for x in score1])
    n_sentence = len(score1)
    sum_score1 = sum([sum(x) for x in score1])
    sum_score2 = sum([sum(x) for x in score2])
    print(f"model: {type(model).__name__}")
    print(f"mean_token_score1: {sum_score1/n_masked}")
    print(f"mean_token_score2: {sum_score2/n_masked}")
    print(f"delta_token_score: {(sum_score1-sum_score2)/n_masked}")
    print(f"mean_sentence_score1: {sum_score1/n_sentence}")
    print(f"mean_sentence_score2: {sum_score2/n_sentence}")
    print(f"delta_sentence_score: {(sum_score1-sum_score2)/n_sentence}")


if __name__ == "__main__":
    main()
