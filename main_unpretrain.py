#!/usr/bin/env python3
# coding:utf-8

# Copyright (c) Tsinghua university conversational AI group (THU-coai).
# This source code is licensed under the MIT license.
import codecs

import torch
import random
from model.utils import load_openai_weights_chinese, set_seed
from model.transformer_model_s2s_soft import TransformerSoftModel
from model.text import myVocab
from config import get_model_config_unpretrain, get_test_config_unpretrain
from collections import Counter
import json
import numpy as np
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
        model_config = get_model_config_unpretrain()
        test_config = get_test_config_unpretrain()
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
        transformer.eval()
        self.model_config = model_config
        self.test_config = test_config
        self.transformer = transformer
        self.device = device
        with open('data/tag2cnt.txt', 'r', encoding='utf8') as fr:
            tags = [line.split('\t')[0] for line in fr.readlines()][:500]
        self.tag2id = {j: i+2 for i, j in enumerate(tags)}
        self.id2tag = {i+2: j for i, j in enumerate(tags)}
        with open('data/loc2cnt.txt', 'r', encoding='utf8') as fr:
            locs = [line.split('\t')[0] for line in fr.readlines()][1:]
        self.loc2id = {j: i+2 for i, j in enumerate(locs)}
        self.id2loc = {i+2: j for i, j in enumerate(locs)}
        self.gender2id = {'男': 1, '女': 2}
        print('Weights loaded from {}'.format(test_config.last_checkpoint_path))

    def next_word_probability(self, context, partial_out, weight_i=None):
        """
        Return probability distribution over next words given a partial true output.
        This is used to calculate the per-word perplexity.

        :param context: dict, contexts containing the dialogue history and personal
                        profile of each speaker
                        this dict contains following keys:

                        context['dialog']: a list of string, dialogue histories (tokens in each utterances
                                           are separated using spaces).
                        context['uid']: a list of int, indices to the profile of each speaker
                        context['profile']: a list of dict, personal profiles for each speaker
                        context['responder_profile']: dict, the personal profile of the responder

        :param partial_out: list, previous "true" words
        :return: a list, the first element is a dict, where each key is a word and each value is a probability
                         score for that word. Unset keys assume a probability of zero.
                         the second element is the probability for the EOS token

        e.g.
        context:
        { "dialog": [ ["How are you ?"], ["I am fine , thank you . And you ?"] ],
          "uid": [0, 1],
          "profile":[ { "loc":"Beijing", "gender":"male", "tag":"" },
                      { "loc":"Shanghai", "gender":"female", "tag":"" } ],
          "responder_profile":{ "loc":"Beijing", "gender":"male", "tag":"" }
        }

        partial_out:
        ['I', 'am']

        ==>  {'fine': 0.9}, 0.1
        """
        '''
        # freqs = copy.deepcopy(self.freqs)
        freqs = self.freqs
        for i in partial_out:
            if i in freqs:
                freqs[i] += 1000
        '''
        if 'responder_profile' in context:
            responder_profile = context['responder_profile']
            tag = responder_profile['tag'].replace(' ', '')
            # weight_i = torch.Tensor([[1, 0]]).to('cuda')
        else:
            responder_profile = context['response_profile']
            tag = ';'.join(responder_profile['tag']).replace(' ', '')
        dialog = context['dialog']
        uid = context['uid']
        profile_all = context['profile']
        # tag = ';'.join(responder_profile['tag']).replace(' ', '')
        loc = ';'.join(responder_profile['loc'].split()).replace(' ', '')
        gender = '男' if responder_profile['gender'] == 'male' else '女'
        persona = '性别:' + gender + ',' + '地点:' + loc + ',' + '标签:' + tag
        profile_ids = self.vocab.string2ids(' '.join(persona))
        dialog_ids = [self.vocab.string2ids(' '.join(i[0].replace(' ', ''))) for i in dialog]
        profile_embedding_ids = []
        for i in profile_all:
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

        profile = [self.vocab.eos_id] + profile_ids + [self.vocab.eos_id]

        history_cat = [self.vocab.eos_id]
        gender_cat = [profile_embedding_ids[uid[0]][0]]
        loc_cat = [profile_embedding_ids[uid[0]][1]]
        tag_cat = [profile_embedding_ids[uid[0]][2]]
        for k in range(len(dialog_ids)):
            temp = dialog_ids[k] + [self.vocab.spl_id]
            history_cat.extend(temp)
            gender_cat.extend([profile_embedding_ids[uid[k]][0]] * len(temp))
            loc_cat.extend([profile_embedding_ids[uid[k]][1]] * len(temp))
            tag_cat.extend([profile_embedding_ids[uid[k]][2]] * len(temp))
        history_cat[-1] = self.vocab.eos_id

        profile = profile[:48]
        history_cat = history_cat[-128:]
        gender_cat = gender_cat[-128:]
        loc_cat = loc_cat[-128:]
        tag_cat = tag_cat[-128:]
        sample = profile, history_cat, gender_cat, loc_cat, tag_cat
        persona, dialog, gender, loc, tag = sample
        contexts = [torch.tensor([c], dtype=torch.long, device=self.device) for c in [persona, dialog] if
                    len(c) > 0]
        gender_format = torch.tensor([gender], dtype=torch.long, device=self.device)
        loc_format = torch.tensor([loc], dtype=torch.long, device=self.device)
        tag_format = torch.tensor([tag], dtype=torch.long, device=self.device)
        with torch.no_grad():
            persona_enc = self.transformer.encode(contexts[0])
            dialog_enc = self.transformer.encode(contexts[1], gender=gender_format, loc=loc_format, tag=tag_format)
            enc_contexts = [persona_enc, dialog_enc]
            if weight_i is None:
                weight = self.transformer.compute_weight(enc_contexts[1])
            else:
                weight = weight_i
            partial_out_ids = self.vocab.string2ids(' '.join(''.join(partial_out)))
            prediction = self.transformer.predict_next(enc_contexts, prefix=partial_out_ids, weight=weight)
        eos_prob = prediction[self.vocab.eos_id]
        distribute = {self.vocab.id2token[i]: max(t, 1e-8) for i, t in enumerate(prediction)}
        return distribute, eos_prob

    def gen_response(self, contexts, weight_i=None):
        """
        Return a list of responses to each context.

        :param contexts: list, a list of context, each context is a dict that contains the dialogue history and personal
                         profile of each speaker
                         this dict contains following keys:

                         context['dialog']: a list of string, dialogue histories (tokens in each utterances
                                            are separated using spaces).
                         context['uid']: a list of int, indices to the profile of each speaker
                         context['profile']: a list of dict, personal profiles for each speaker
                         context['responder_profile']: dict, the personal profile of the responder

        :return: list, responses for each context, each response is a list of tokens.

        e.g.
        contexts:
        [{ "dialog": [ ["How are you ?"], ["I am fine , thank you . And you ?"] ],
          "uid": [0, 1],
          "profile":[ { "loc":"Beijing", "gender":"male", "tag":"" },
                      { "loc":"Shanghai", "gender":"female", "tag":"" } ],
          "responder_profile":{ "loc":"Beijing", "gender":"male", "tag":"" }
        }]

        ==>  [['I', 'am', 'fine', 'too', '!']]
        """
        res = []
        for context in contexts:
            if 'responder_profile' in context:
                responder_profile = context['responder_profile']
                tag = responder_profile['tag'].replace(' ', '')
                #weight_i = torch.Tensor([[1, 0]]).to('cuda')
            else:
                responder_profile = context['response_profile']
                tag = ';'.join(responder_profile['tag']).replace(' ', '')
            dialog = context['dialog']
            uid = context['uid']
            profile_all = context['profile']
            # tag = ';'.join(responder_profile['tag']).replace(' ', '')
            loc = ';'.join(responder_profile['loc'].split()).replace(' ', '')
            gender = '男' if responder_profile['gender'] == 'male' else '女'
            persona = '性别:' + gender + ',' + '地点:' + loc + ',' + '标签:' + tag
            profile_ids = self.vocab.string2ids(' '.join(persona))
            dialog_ids = [self.vocab.string2ids(' '.join(i[0].replace(' ', ''))) for i in dialog]
            profile_embedding_ids = []
            for i in profile_all:
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

            profile = [self.vocab.eos_id] + profile_ids + [self.vocab.eos_id]

            history_cat = [self.vocab.eos_id]
            gender_cat = [profile_embedding_ids[uid[0]][0]]
            loc_cat = [profile_embedding_ids[uid[0]][1]]
            tag_cat = [profile_embedding_ids[uid[0]][2]]
            for k in range(len(dialog_ids)):
                temp = dialog_ids[k] + [self.vocab.spl_id]
                history_cat.extend(temp)
                gender_cat.extend([profile_embedding_ids[uid[k]][0]] * len(temp))
                loc_cat.extend([profile_embedding_ids[uid[k]][1]] * len(temp))
                tag_cat.extend([profile_embedding_ids[uid[k]][2]] * len(temp))
            history_cat[-1] = self.vocab.eos_id

            profile = profile[:48]
            history_cat = history_cat[-128:]
            gender_cat = gender_cat[-128:]
            loc_cat = loc_cat[-128:]
            tag_cat = tag_cat[-128:]
            sample = profile, history_cat, gender_cat, loc_cat, tag_cat
            persona, dialog, gender, loc, tag = sample
            contexts = [torch.tensor([c], dtype=torch.long, device=self.device) for c in [persona, dialog] if
                        len(c) > 0]
            gender_format = torch.tensor([gender], dtype=torch.long, device=self.device)
            loc_format = torch.tensor([loc], dtype=torch.long, device=self.device)
            tag_format = torch.tensor([tag], dtype=torch.long, device=self.device)
            with torch.no_grad():
                persona_enc = self.transformer.encode(contexts[0])
                dialog_enc = self.transformer.encode(contexts[1], gender=gender_format, loc=loc_format, tag=tag_format)
                enc_contexts = [persona_enc, dialog_enc]
                if weight_i is None:
                    weight = self.transformer.compute_weight(enc_contexts[1])
                else:
                    weight = weight_i
                prediction = self.transformer.beam_search(enc_contexts, weight=weight)[0]
            prediction_str = self.vocab.ids2string(prediction)
            res.append(list(prediction_str))
        return res

def test_biased(model):
    with open('data/test_data_biased_unpretrain.txt', 'w', encoding='utf8') as fw:
        with open('data/test_data_biased.json', 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            weights = [torch.Tensor([[1, 0]]), torch.Tensor([[0, 1]])]
            weights = [i.to('cuda') for i in weights]
            for line in lines:
                line = line.strip('\n')
                data = json.loads(line)
                dialog = data['dialog']
                uid = data['uid']
                profile_all = data['profile']
                responder_profile = data['responder_profile']
                golden_response = data['golden_response']
                golden_response_str = ''.join(golden_response).replace(' ', '')
                dialog_str = '\n\t'.join([''.join(i).replace(' ', '') for i in dialog])
                profile_all_str = '\n\t'.join([json.dumps(i, ensure_ascii=False) for i in profile_all])
                responder_profile_str = json.dumps(responder_profile, ensure_ascii=False)
                fw.write('all profiles: \n\t' + profile_all_str + '\n')
                fw.write('responder profile: \n\t' + responder_profile_str + '\n')
                fw.write('history: \n\t' + dialog_str + '\n')
                fw.write('golden response: \n\t' + golden_response_str + '\n')
                ans_auto = model.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                ans_persona = model.gen_response([data],weight_i=weights[0])
                ans_persona = ''.join(ans_persona[0])
                ans_nopersona = model.gen_response([data], weight_i=weights[1])
                ans_nopersona = ''.join(ans_nopersona[0])
                fw.write('predict with auto weight: ' + ans_auto + '\n')
                fw.write('predict with full persona: ' + ans_persona + '\n')
                fw.write('predict with no persona: ' + ans_nopersona + '\n')
                fw.write('\n')

def test_random(model):
    with open('data/test_data_random_unpretrain.txt', 'w', encoding='utf8') as fw:
        with open('data/test_data_random.json', 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            weights = [torch.Tensor([[1, 0]]), torch.Tensor([[0, 1]])]
            weights = [i.to('cuda') for i in weights]
            for line in lines:
                line = line.strip('\n')
                data = json.loads(line)
                dialog = data['dialog']
                uid = data['uid']
                profile_all = data['profile']
                responder_profile = data['response_profile']
                golden_response = data['golden_response']
                golden_response_str = ''.join(golden_response).replace(' ', '')
                dialog_str = '\n\t'.join([''.join(i).replace(' ', '') for i in dialog])
                profile_all_str = '\n\t'.join([json.dumps(i, ensure_ascii=False) for i in profile_all])
                responder_profile_str = json.dumps(responder_profile, ensure_ascii=False)
                fw.write('all profiles: \n\t' + profile_all_str + '\n')
                fw.write('responder profile: \n\t' + responder_profile_str + '\n')
                fw.write('history: \n\t' + dialog_str + '\n')
                fw.write('golden response: \n\t' + golden_response_str + '\n')
                ans_auto = model.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                ans_persona = model.gen_response([data],weight_i=weights[0])
                ans_persona = ''.join(ans_persona[0])
                ans_nopersona = model.gen_response([data], weight_i=weights[1])
                ans_nopersona = ''.join(ans_nopersona[0])
                fw.write('predict with auto weight: ' + ans_auto + '\n')
                fw.write('predict with full persona: ' + ans_persona + '\n')
                fw.write('predict with no persona: ' + ans_nopersona + '\n')
                fw.write('\n')

if __name__ == '__main__':
    model = Model()
    # test_biased(model)
    test_random(model)


