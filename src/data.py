from dataclasses import dataclass
import os
import copy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def extract_answer(text):
    split_pattern = '####'
    if split_pattern not in text:
        # return text.strip().replace(',', '')
        return text
    else:
        _, ans = text.strip().split('####', 1)
        ans = '####' + ans
        # ans = ans.strip().replace(',', '')
        return ans

def extract_cot(text):
    split_pattern = '####'
    if split_pattern not in text:
        return None
    else:
        cot, _ = text.strip().split('####', 1)
        cot = cot.strip()
        return cot

class CoTDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        print (f'Creating features from dataset file at {file_path}')
        self.tokenizer = tokenizer
        self.max_length = max_length
        bos_tok = tokenizer.bos_token
        eos_tok = tokenizer.eos_token

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                             and len(line.split('||')) ==2 )]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        edited_sents_cot = []
        edited_sents_only = []
        edited_sents_all = []
        edited_sents_nocot = []
        for src, tgt in zip(src_lines, tgt_lines):
            #import pdb; pdb.set_trace()
            ans = extract_answer(tgt)
            cot = extract_cot(tgt)
            sent = ' {} {} '.format(src, bos_tok) + cot + ' {}'.format(eos_tok)
            edited_sents_cot.append(sent)
            sent = ' {} {} '.format(src, bos_tok)
            edited_sents_only.append(sent)
            sent = ' {} {} '.format(src, bos_tok) + cot + ' {} '.format(eos_tok) + ans + ' {}'.format(eos_tok)
            edited_sents_all.append(sent)
            sent = ' {} {} '.format(src, bos_tok) + ans + ' {}'.format(eos_tok)
            edited_sents_nocot.append(sent)
        

        self.edited_sents_cot = edited_sents_cot
        self.edited_sents_only = edited_sents_only
        self.edited_sents_all = edited_sents_all
        self.edited_sents_nocot = edited_sents_nocot

        # batch_encoding_cot = tokenizer(edited_sents_cot, add_special_tokens=True, truncation=True, max_length=max_length)
        # batch_encoding_only = tokenizer(edited_sents_only, add_special_tokens=True, truncation=True, max_length=max_length)
        # batch_encoding_all = tokenizer(edited_sents_all, add_special_tokens=True, truncation=True, max_length=max_length)
        # batch_encoding_nocot = tokenizer(edited_sents_nocot, add_special_tokens=True, truncation=True, max_length=max_length)
        # self.examples_cot = batch_encoding_cot["input_ids"]
        # self.examples_only = batch_encoding_only["input_ids"]
        # self.examples_all = batch_encoding_all["input_ids"]
        # self.examples_nocot = batch_encoding_nocot["input_ids"]

        # self.labels_cot = copy.deepcopy(self.examples_cot)
        # self.labels_all = copy.deepcopy(self.examples_all)
        # self.labels_cot_shift = copy.deepcopy(self.examples_cot)
        # self.labels_nocot = copy.deepcopy(self.examples_nocot)

        # self.src_sent_cot = []
        # self.tgt_sent_cot = []

        # temp_src_len = 0
        # temp_tgt_len = 0
        # temp_count = 0
        # separator = tokenizer.eos_token_id #tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        # for i, elem in enumerate(self.labels_cot):
        #     sep_idx = elem.index(separator) + 1
        #     self.src_sent_cot.append(self.examples_cot[i][:sep_idx-1])
        #     self.tgt_sent_cot.append(self.examples_cot[i][sep_idx-1:])
        #     self.labels_cot[i][:sep_idx] = [-100] * sep_idx
        #     assert self.labels_all[i][sep_idx-1] == separator
        #     self.labels_all[i][:sep_idx] = [-100] * sep_idx
        #     self.labels_cot_shift[i][:sep_idx-1] = [-100] * (sep_idx-1)
        #     temp_src_len += sep_idx-1
        #     temp_tgt_len += len(elem) - (sep_idx-1)
        #     temp_count += 1

        # print('tgt_avg: ', temp_tgt_len / temp_count)
        # print('src_avg: ', temp_src_len / temp_count)
        # print('ratios: ', temp_src_len/temp_tgt_len)

        # self.src_sent_nocot = []
        # self.tgt_sent_nocot = []
        # temp_src_len = 0
        # temp_tgt_len = 0
        # temp_count = 0
        # separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        # for i, elem in enumerate(self.labels_nocot):
        #     sep_idx = elem.index(separator) + 1
        #     self.src_sent_nocot.append(self.examples_nocot[i][:sep_idx-1])
        #     self.tgt_sent_nocot.append(self.examples_nocot[i][sep_idx-1:])
        #     self.labels_nocot[i][:sep_idx] = [-100] * sep_idx
        #     temp_src_len += sep_idx-1
        #     temp_tgt_len += len(elem) - (sep_idx-1)
        #     temp_count += 1

        # print('tgt_avg: ', temp_tgt_len / temp_count)
        # print('src_avg: ', temp_src_len / temp_count)
        # print('ratios: ', temp_src_len/temp_tgt_len)


        # print(edited_sents_all[0])
        # print(self.labels_cot[0])
        # print(self.labels_nocot[0])
        # print(self.examples_nocot[0])
        # print(edited_sents_nocot[0])
        # print(self.src_sent_nocot[0])
        # print(self.tgt_sent_nocot[0])

    def __len__(self):
        return len(self.examples_cot)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):

        
        bos_tok = self.tokenizer.bos_token

        batch_encoding_cot = self.tokenizer([self.edited_sents_cot[i]], add_special_tokens=True, truncation=True, max_length=self.max_length)[0]
        batch_encoding_only = self.tokenizer([self.edited_sents_only[i]], add_special_tokens=True, truncation=True, max_length=self.max_length)[0]
        batch_encoding_all = self.tokenizer([self.edited_sents_all[i]], add_special_tokens=True, truncation=True, max_length=self.max_length)[0]
        batch_encoding_nocot = self.tokenizer([self.edited_sents_nocot[i]], add_special_tokens=True, truncation=True, max_length=self.max_length)[0]

        examples_cot = batch_encoding_cot["input_ids"]
        examples_only = batch_encoding_only["input_ids"]
        examples_all = batch_encoding_all["input_ids"]
        examples_nocot = batch_encoding_nocot["input_ids"]

        labels_cot = copy.deepcopy(examples_cot)
        labels_all = copy.deepcopy(examples_all)
        labels_cot_shift = copy.deepcopy(examples_cot)
        labels_nocot = copy.deepcopy(examples_nocot)

        src_sent_cot = None
        tgt_sent_cot = None

        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        separator = self.tokenizer.eos_token_id #tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        
        elem = labels_cot
        sep_idx = elem.index(separator) + 1
        src_sent_cot = examples_cot[:sep_idx-1]
        tgt_sent_cot = examples_cot[sep_idx-1:]
        labels_cot[:sep_idx] = [-100] * sep_idx
        assert labels_all[sep_idx-1] == separator
        labels_all[:sep_idx] = [-100] * sep_idx
        labels_cot_shift[:sep_idx-1] = [-100] * (sep_idx-1)

        src_sent_nocot = None
        tgt_sent_nocot = None
        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        separator = self.tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]

        elem = labels_nocot
        sep_idx = elem.index(separator) + 1
        src_sent_nocot = examples_nocot[:sep_idx-1]
        tgt_sent_nocot = examples_nocot[sep_idx-1:]
        labels_nocot[:sep_idx] = [-100] * sep_idx




        return (torch.tensor(examples_cot, dtype=torch.long),
                torch.tensor(examples_nocot, dtype=torch.long),
                torch.tensor(labels_cot, dtype=torch.long),
                torch.tensor(labels_cot_shift, dtype=torch.long),
                torch.tensor(labels_nocot, dtype=torch.long),
                torch.tensor(src_sent_cot, dtype=torch.long),
                torch.tensor(src_sent_nocot, dtype=torch.long),
                torch.tensor(tgt_sent_cot, dtype=torch.long),
                torch.tensor(tgt_sent_nocot, dtype=torch.long),
                torch.tensor(examples_only, dtype=torch.long),
                torch.tensor(examples_all, dtype=torch.long),
                torch.tensor(labels_all, dtype=torch.long),
                )
@dataclass
class CoTDataCollator:
    """
    VAEData collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        #import pdb; pdb.set_trace()
        input_ids_cot, input_ids_nocot, labels_cot, labels_cot_shift, labels_nocot, src_cot, src_nocot, tgt_cot, tgt_nocot, input_ids_only, input_ids_all, labels_all = zip(*examples)
        input_ids_cot = self._tensorize_batch(input_ids_cot)
        input_ids_cot[input_ids_cot.lt(0)] = self.tokenizer.eos_token_id
        input_ids_only = self._tensorize_batch(input_ids_only)
        input_ids_only[input_ids_only.lt(0)] = self.tokenizer.eos_token_id
        input_ids_all = self._tensorize_batch(input_ids_all)
        input_ids_all[input_ids_all.lt(0)] = self.tokenizer.eos_token_id
        input_ids_nocot = self._tensorize_batch(input_ids_nocot)
        input_ids_nocot[input_ids_nocot.lt(0)] = self.tokenizer.eos_token_id
        labels_cot = self._tensorize_batch(labels_cot)
        labels_all = self._tensorize_batch(labels_all)
        labels_cot_shift = self._tensorize_batch(labels_cot_shift)
        labels_nocot = self._tensorize_batch(labels_nocot)
        return {"input_ids_cot": input_ids_cot, "input_ids_nocot": input_ids_nocot, "labels_cot": labels_cot, "labels_cot_shift": labels_cot_shift, "labels_nocot": labels_nocot, 'input_ids_only': input_ids_only, 'input_ids_all': input_ids_all, 'labels_all': labels_all}

    def _tensorize_batch(self, examples):
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=-100)
