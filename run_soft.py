import torch
import random
from model.utils import load_openai_weights_chinese, set_seed
from model.transformer_model_s2s_soft import TransformerSoftModel
from model.text import myVocab
from config import get_model_config_soft, get_test_config_soft
from collections import Counter
import json
import warnings

warnings.filterwarnings("ignore")


class Model:
    """
    This is an example model. It reads predefined dictionary and predict a fixed distribution.
    For a correct evaluation, each team should implement 3 functions:
    next_word_probability
    gen_response
    """

    def __init__(self):
        """
        Init whatever you need here
        with codecs.open(vocab_file, 'r', 'utf-8') as f:
            vocab = [i.strip().split()[0] for i in f.readlines() if len(i.strip()) != 0]
        self.vocab = vocab
        self.freqs = dict(zip(self.vocab[::-1], range(len(self.vocab))))
        """
        # vocab_file = 'vocab.txt'
        model_config = get_model_config_soft()
        test_config = get_test_config_soft()
        set_seed(test_config.seed)
        device = torch.device(test_config.device)
        vocab = myVocab(model_config.vocab_path)
        self.vocab = vocab
        transformer = TransformerSoftModel(n_layers=model_config.n_layers,
                                           n_embeddings=len(vocab),
                                           n_pos_embeddings=model_config.n_pos_embeddings,
                                           embeddings_size=model_config.embeddings_size,
                                           padding_idx=vocab.pad_id,
                                           n_heads=model_config.n_heads,
                                           dropout=model_config.dropout,
                                           embed_dropout=model_config.embed_dropout,
                                           attn_dropout=model_config.attn_dropout,
                                           ff_dropout=model_config.ff_dropout,
                                           bos_id=vocab.bos_id,
                                           eos_id=vocab.eos_id,
                                           max_seq_len=model_config.max_seq_len,
                                           beam_size=model_config.beam_size,
                                           length_penalty=model_config.length_penalty,
                                           n_segments=model_config.n_segments,
                                           annealing_topk=model_config.annealing_topk,
                                           temperature=model_config.temperature,
                                           annealing=model_config.annealing,
                                           diversity_coef=model_config.diversity_coef,
                                           diversity_groups=model_config.diversity_groups,
                                           n_gender=3,
                                           n_loc=37,
                                           n_tag=502)
        transformer = transformer.to(device)
        state_dict = torch.load(test_config.last_checkpoint_path, map_location=device)
        temp = dict(state_dict['model'])
        keys = list(temp.keys())
        for key in keys:
            # new_key = '.'.join([i for i in key.split('.') if i != 'module'])
            new_key = key.replace('.module', '')
            temp[new_key] = temp.pop(key)
        transformer.load_state_dict(temp)
        # transformer.load_state_dict(state_dict['model'])
        transformer.eval()
        self.model_config = model_config
        self.test_config = test_config
        self.transformer = transformer
        self.device = device
        with open('data/tag2cnt.txt', 'r', encoding='utf8') as fr:
            tags = [line.split('\t')[0] for line in fr.readlines()][:500]
        self.tag2id = {j: i + 2 for i, j in enumerate(tags)}
        self.id2tag = {i + 2: j for i, j in enumerate(tags)}
        with open('data/loc2cnt.txt', 'r', encoding='utf8') as fr:
            locs = [line.split('\t')[0] for line in fr.readlines()][1:]
        self.loc2id = {j: i + 2 for i, j in enumerate(locs)}
        self.id2loc = {i + 2: j for i, j in enumerate(locs)}
        self.gender2id = {'男': 1, '女': 2}
        print('Weights loaded from {}'.format(test_config.last_checkpoint_path))

    def gen_response(self, line, weight_i=None):
        label, _, line = line.strip('\n').split('\t')
        label = int(label)
        data = json.loads(line)
        dialog = data['dialog']
        profile = data['profile']
        uid = data['uid']
        dialog_ids = [self.vocab.string2ids(' '.join(i[0].replace(' ', ''))) for i in dialog]
        profile_ids = []
        profile_embedding_ids = []
        for i in profile:
            temp = []
            tag = ';'.join(i['tag']).replace(' ', '')
            loc = ';'.join(i['loc'].split()).replace(' ', '')
            gender = '男' if i['gender'] == 'male' else '女'
            temp.append(self.gender2id[gender])
            loc_list = loc.split(';')
            if len(loc_list):
                if loc_list[0] in self.loc2id:
                    temp.append(self.loc2id[loc_list[0]])
                else:
                    temp.append(1)
            else:
                temp.append(1)
            tag_list = tag.split(';')
            for j in tag_list:
                if j in self.tag2id:
                    temp.append(self.tag2id[j])
                    break
            if len(temp) == 2:
                temp.append(1)
            profile_embedding_ids.append(temp)
            persona = '性别:' + gender + ',' + '地点:' + loc + ',' + '标签:' + tag
            profile_ids.append(self.vocab.string2ids(' '.join(persona)))
        # sample = dialog_ids, profile_ids, uid, profile_embedding_ids, label
        # dialog_ids, profile_ids, uid, profile_embedding_ids, label = sample
        # **************
        l = len(dialog_ids)
        n = random.choice(range(1, l))
        profile = profile_ids[uid[n]]
        # profile_embedding = profile_embedding_ids[uid[n]]
        history = dialog_ids[:n]
        y = dialog_ids[n]
        profile = [self.vocab.eos_id] + profile + [self.vocab.eos_id]
        history_cat = [self.vocab.eos_id]
        gender_cat = [profile_embedding_ids[uid[0]][0]]
        loc_cat = [profile_embedding_ids[uid[0]][1]]
        tag_cat = [profile_embedding_ids[uid[0]][2]]
        for k in range(n):
            temp = history[k] + [self.vocab.spl_id]
            history_cat.extend(temp)
            gender_cat.extend([profile_embedding_ids[uid[k]][0]] * len(temp))
            loc_cat.extend([profile_embedding_ids[uid[k]][1]] * len(temp))
            tag_cat.extend([profile_embedding_ids[uid[k]][2]] * len(temp))
        history_cat[-1] = self.vocab.eos_id
        y = [self.vocab.eos_id] + y + [self.vocab.eos_id]
        profile = profile[:48]
        history_cat = history_cat[-128:]
        gender_cat = gender_cat[-128:]
        loc_cat = loc_cat[-128:]
        tag_cat = tag_cat[-128:]
        y = y[:32]
        sample = profile, history_cat, y, gender_cat, loc_cat, tag_cat, label
        persona, dialog, target, gender, loc, tag, label = sample

        contexts = [torch.tensor([c], dtype=torch.long, device=self.device) for c in [persona, dialog] if
                    len(c) > 0]
        gender_format = torch.tensor([gender], dtype=torch.long, device=self.device)
        loc_format = torch.tensor([loc], dtype=torch.long, device=self.device)
        tag_format = torch.tensor([tag], dtype=torch.long, device=self.device)
        persona_enc = self.transformer.encode(contexts[0])
        dialog_enc = self.transformer.encode(contexts[1], gender=gender_format, loc=loc_format, tag=tag_format)
        enc_contexts = [persona_enc, dialog_enc]
        if weight_i is None:
            weight = self.transformer.compute_weight(enc_contexts[1])
        else:
            weight = weight_i
        prediction = self.transformer.beam_search(enc_contexts, weight=weight)[0]
        context_str = self.vocab.ids2string(persona[1:-1])
        dialog_str = self.vocab.ids2string(dialog)
        dialog_str = dialog_str.replace(self.vocab.spl, '\n\t- ')
        target_str = self.vocab.ids2string(target[1:-1])
        prediction_str = self.vocab.ids2string(prediction)
        weight = '\t'.join([str(i) for i in weight.tolist()[0]])
        '''
        print('\n')
        print('weight: ', weight)
        print('label: ', label)
        print('Persona info:\n\t{}'.format(context_str))
        print('Dialog:{}'.format(dialog_str))
        print('Target:\n\t{}'.format(target_str))
        print('Prediction:\n\t{}'.format(prediction_str))
        '''
        return weight, str(label), context_str, dialog_str, target_str, prediction_str


if __name__ == '__main__':
    model = Model()
    import tqdm

    with open('data/valid_all_100_predict_weight.txt', 'w', encoding='utf8') as fw:
        with open('data/soft/test_data_label.json', 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            weights = [torch.Tensor([[1, 0]]), torch.Tensor([[0, 1]])]
            weights = [i.to('cuda') for i in weights]
            for line in tqdm.tqdm(lines[:100]):  ##TODO change the nums
                temp = []
                temp_weight = []
                weight, label, context_str, dialog_str, target_str, prediction_str = model.gen_response(line, weight_i=
                weights[0])
                temp.append(prediction_str)
                temp_weight.append(weight)
                weight, label, context_str, dialog_str, target_str, prediction_str = model.gen_response(line, weight_i=
                weights[1])
                temp.append(prediction_str)
                temp_weight.append(weight)
                weight, label, context_str, dialog_str, target_str, prediction_str = model.gen_response(line,
                                                                                                        weight_i=None)
                temp.append(prediction_str)
                temp_weight.append(weight)
                fw.write(line)
                fw.write('weight: ' + 'persons: ' + temp_weight[0] + ', no persona: ' + temp_weight[1] + ', soft: ' +
                         temp_weight[2] + '\n')
                fw.write('label: ' + label + '\n')
                fw.write('Persona info:\n\t{}'.format(context_str) + '\n')
                fw.write('Dialog:{}'.format(dialog_str) + '\n')
                fw.write('Target:\n\t{}'.format(target_str) + '\n')
                fw.write(
                    'Prediction:\n\t' + 'persona: ' + temp[0] + '\n\t' + 'no persona: ' + temp[1] + '\n\t' + 'soft: ' +
                    temp[2] + '\n')
                fw.write('\n')
