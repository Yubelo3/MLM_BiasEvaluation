import csv
from torch.utils.data import Dataset
import difflib


class CrowSDataset(Dataset):
    # ['', 'sent_more', 'sent_less', 'stereo_antistereo', 'bias_type', 'annotations', 'anon_writer', 'anon_annotators']
    # {'physical-appearance', 'religion', 'socioeconomic', 'disability', 'sexual-orientation', 'age', 'race-color', 'nationality', 'gender'}
    def __init__(
        self,
        filepath="crows_pairs_anonymized.csv",
        type_filter=["physical-appearance"]
    ) -> None:
        super().__init__()
        self.sent_more=[]
        self.sent_less=[]
        with open(filepath,"r") as f:
            reader=csv.DictReader(f)
            for row in reader:
                if row["bias_type"] not in type_filter:
                    continue
                assert row["stereo_antistereo"] in ["stereo","antistereo"]
                if row["stereo_antistereo"]=="stereo":
                    self.sent_more.append(row["sent_more"])
                    self.sent_less.append(row["sent_less"])
                else:
                    self.sent_more.append(row["sent_less"])
                    self.sent_less.append(row["sent_more"])
    
    def __len__(self):
        return len(self.sent_more)
    
    def __getitem__(self,index):
        return [self.sent_more[index],self.sent_less[index]]






if __name__=="__main__":
    dataset=CrowSDataset()
    print(dataset[0])
    print(dataset[5])