#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_module_soft import TransformerModule


class TransformerUnweightModel(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size,
                 padding_idx, n_heads, dropout, embed_dropout, attn_dropout, ff_dropout,
                 bos_id, eos_id, max_seq_len=256, beam_size=5, sample=False,
                 length_penalty=0.8, annealing_topk=None, temperature=0.8, annealing=0,
                 diversity_coef=0, diversity_groups=1, n_segments=None, n_gender=3, n_loc=37, n_tag=502):

        super(TransformerUnweightModel, self).__init__()

        self.padding_idx = padding_idx
        self.n_embeddings = n_embeddings
        self.n_pos_embeddings = n_pos_embeddings
        self.embeddings_size = embeddings_size

        self.bos_id = bos_id
        self.eos_id = eos_id

        self.max_seq_len = max_seq_len
        self.beam_size = beam_size
        self.sample = sample
        self.length_penalty_coef = length_penalty
        self.annealing = annealing
        self.annealing_topk = annealing_topk
        self.temperature = temperature
        self.diversity_coef = diversity_coef
        self.diversity_groups = diversity_groups

        self.transformer_module = TransformerModule(n_layers, n_embeddings, n_pos_embeddings, embeddings_size,
                                                    padding_idx, n_heads, dropout, embed_dropout, attn_dropout,
                                                    ff_dropout, n_segments, n_gender=n_gender, n_loc=n_loc, n_tag=n_tag)
        self.pre_softmax = nn.Linear(embeddings_size, n_embeddings, bias=False)
        self.pre_softmax.weight = self.transformer_module.embeddings.weight
        # self.cls_linear = nn.Linear(embeddings_size, 2, bias=False)

    def forward(self, x, contexts=[]):
        enc_contexts = [self.encode(c) for c in contexts]
        return self.decode(x, enc_contexts)

    def encode(self, x, gender=None, loc=None, tag=None):
        return self.transformer_module(x, gender=gender, loc=loc, tag=tag)

    def generate(self, enc_x):
        return self.pre_softmax(enc_x)

    def classify(self, enc_x):
        # print(enc_x[0])
        # print(enc_x[0].shape)
        output = enc_x[0]  # b*l*dim
        mask = enc_x[1]   # b*l
        one_mask = ~mask
        output = output * one_mask.float().unsqueeze(2)
        length = torch.sum(one_mask, 1).unsqueeze(1)
        output = torch.sum(output, 1) # b*dim
        avg_output = output/length.float()
        return self.cls_linear(avg_output)

    def compute_weight(self, enc_x):
        # print(enc_x[0])
        # print(enc_x[0].shape)
        output = enc_x[0]  # b*l*dim
        mask = enc_x[1]   # b*l
        one_mask = ~mask
        output = output * one_mask.float().unsqueeze(2)
        length = torch.sum(one_mask, 1).unsqueeze(1)
        output = torch.sum(output, 1) # b*dim
        avg_output = output/length.float()
        weight = F.softmax(self.cls_linear(avg_output), dim=-1)
        return weight

    def decode(self, x, enc_contexts=[], weight=None):
        x, _ = self.transformer_module(x, enc_contexts, weight=weight)
        return self.generate(x)

    def predict(self, contexts=[]):
        enc_contexts = [self.encode(c) for c in contexts]
        prediction = self.beam_search(enc_contexts)
        return prediction

    def predict_beam(self, contexts=[]):
        enc_contexts = [self.encode(c) for c in contexts]
        prediction = self.beam_search(enc_contexts, return_beams=True)

        return prediction

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.length_penalty_coef / (5 + 1) ** self.length_penalty_coef

    def predict_next(self, enc_contexts=[], return_beams=False, prefix=[], weight=None):
        with torch.no_grad():
            if len(enc_contexts) == 0:
                return []

            batch_size = enc_contexts[0][0].shape[0]
            device = next(self.parameters()).device

            ind = len(prefix)
            if ind:
                assert batch_size == 1
                prefix_sentence = [self.bos_id] + prefix
                prevs = torch.LongTensor(prefix_sentence).to(device)
                prevs = prevs.expand(self.beam_size, ind + 1)
            else:
                prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.bos_id, dtype=torch.long,
                                   device=device)
            beam_enc_contexts = []
            for c, p in enc_contexts:
                c = c.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
                c = c.view(-1, c.shape[2], c.shape[3])
                p = p.unsqueeze(1).repeat(1, self.beam_size, 1)
                p = p.view(-1, p.shape[2])
                beam_enc_contexts.append((c, p))
            # zrs
            if weight is not None:
                weight = weight.unsqueeze(1).repeat(1, self.beam_size, 1)
                weight = weight.view(-1, weight.shape[2])
            outputs, _ = self.transformer_module(prevs, beam_enc_contexts, weight=weight)
            logits = self.generate(outputs[:, -1, :])
            probs = F.softmax(logits, dim=-1)
        return probs[0].tolist()

    def beam_search(self, enc_contexts=[], return_beams=False, weight=None):
        with torch.no_grad():
            if len(enc_contexts) == 0:
                return []

            batch_size = enc_contexts[0][0].shape[0]
            device = next(self.parameters()).device

            prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.bos_id, dtype=torch.long,
                               device=device)

            beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
            beam_lens = torch.ones(batch_size, self.beam_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size, self.beam_size, dtype=torch.uint8, device=device)

            beam_enc_contexts = []
            for c, p in enc_contexts:
                c = c.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
                c = c.view(-1, c.shape[2], c.shape[3])
                p = p.unsqueeze(1).repeat(1, self.beam_size, 1)
                p = p.view(-1, p.shape[2])
                beam_enc_contexts.append((c, p))

            # zrs
            if weight is not None:
                weight = weight.unsqueeze(1).repeat(1, self.beam_size, 1)
                weight = weight.view(-1, weight.shape[2])

            current_sample_prob = 1
            group_size = self.beam_size // self.diversity_groups
            diversity_penalty = torch.zeros((batch_size, self.n_embeddings), device=device)

            # zrs:
            repeat = [{} for i in range(batch_size * self.beam_size)]
            # **********
            for i in range(self.max_seq_len):
                outputs, _ = self.transformer_module(prevs, beam_enc_contexts, weight=weight)

                logits = self.generate(outputs[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
                # zrs: remove n repeat. prevs: (batch_size*beam_size, 1)
                for idx in range(batch_size * self.beam_size):
                    for key in repeat[idx]:
                        for value in repeat[idx][key]:
                            log_probs[idx][value] = -1000
                # **********
                log_probs = log_probs.view(batch_size, self.beam_size, -1)

                beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1))
                # zrs, log_probs: batch * beam * dim
                ba, be, dim = beam_scores.shape
                for ba_idx in range(ba):
                    for be_idx in range(be):
                        if int(torch.max(beam_scores[ba_idx][be_idx]) == torch.min(beam_scores[ba_idx][be_idx])):
                            temp = float(beam_scores[ba_idx][be_idx][0])
                            beam_scores[ba_idx][be_idx] = -float('inf')
                            beam_scores[ba_idx][be_idx][0] = temp
                # **********
                penalty = self._length_penalty(beam_lens.float() + 1 - is_end.float())
                penalty = penalty.unsqueeze(-1).repeat(1, 1, self.n_embeddings)
                beam_scores = beam_scores / penalty

                if i == 0:
                    penalty = penalty[:, 0, :]
                    beam_scores = beam_scores[:, 0, :]

                    beam_scores, idxs = beam_scores.topk(self.beam_size, dim=-1)
                    beam_idxs = torch.zeros((batch_size, self.beam_size), dtype=torch.long, device=device)
                else:

                    penalty = penalty.view(batch_size, self.diversity_groups, group_size, -1)
                    beam_scores = beam_scores.view(batch_size, self.diversity_groups, group_size, -1)

                    all_scores, all_idxs = [], []
                    for g in range(self.diversity_groups):
                        g_beam_scores = beam_scores[:, g, :, :]
                        g_penalty = penalty[:, g, :, :]
                        g_beam_scores -= self.diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
                        g_beam_scores = g_beam_scores.view(batch_size, -1)

                        if random.random() < current_sample_prob:
                            # print('*********')
                            beam_probas = F.softmax(g_beam_scores/self.temperature, dim=-1)
                            if self.annealing_topk is not None:
                                beam_probas, sample_idxs = beam_probas.topk(self.annealing_topk, dim=-1)
                                g_idxs = torch.multinomial(beam_probas, group_size)
                                g_idxs = torch.gather(sample_idxs, 1, g_idxs)
                            else:
                                g_idxs = torch.multinomial(beam_probas, group_size)
                        else:
                            # print('|||||||||')
                            _, g_idxs = g_beam_scores.topk(group_size, dim=-1)

                        g_scores = torch.gather(beam_scores[:, g, :, :].view(batch_size, -1), 1, g_idxs)
                        g_idxs += g * group_size * self.n_embeddings

                        all_scores.append(g_scores)
                        all_idxs.append(g_idxs)

                        diversity_penalty.scatter_add_(1, torch.fmod(g_idxs, self.n_embeddings),
                                                       torch.ones((batch_size, group_size), device=device))

                    diversity_penalty.fill_(0).byte()
                    penalty = penalty.view(batch_size, -1)
                    beam_scores = torch.cat(all_scores, dim=-1)
                    idxs = torch.cat(all_idxs, dim=-1)

                    beam_idxs = (idxs.float() / self.n_embeddings).long()

                penalty = torch.gather(penalty, 1, idxs)
                sym_idxs = torch.fmod(idxs, log_probs.shape[-1])
                is_end = torch.gather(is_end, 1, beam_idxs)
                beam_lens = torch.gather(beam_lens, 1, beam_idxs)

                sym_idxs[is_end] = self.padding_idx
                beam_lens[~is_end] += 1
                is_end[sym_idxs == self.eos_id] = 1

                sym_idxs = sym_idxs.view(batch_size * self.beam_size, 1)
                prevs = prevs.view(batch_size, self.beam_size, -1)
                prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
                prevs = prevs.view(batch_size * self.beam_size, -1)
                prevs = torch.cat([prevs, sym_idxs], dim=1)

                # zrs:
                prevs_list = prevs.tolist()
                for b in range(batch_size * self.beam_size):
                    b_list = prevs_list[b]
                    if len(b_list) > 2 and b_list[-1] != self.padding_idx and b_list[-1] != self.eos_id:
                        key = (int(b_list[-3]), int(b_list[-2]))
                        if key in repeat[b]:
                            repeat[b][key].append(int(b_list[-1]))
                        else:
                            repeat[b][key] = [int(b_list[-1])]
                # ********

                if all(is_end.view(-1)):
                    break

                beam_scores *= penalty
                current_sample_prob *= self.annealing

            predicts = []
            result = prevs.view(batch_size, self.beam_size, -1)

            if return_beams:
                bests = torch.argsort(beam_scores, dim=-1, descending=True)
                for i in range(batch_size):
                    temp = []
                    for j in range(self.beam_size):
                        best_len = beam_lens[i, bests[i][j]]
                        best_seq = result[i, bests[i][j], 1:best_len - 1]
                        temp.append(best_seq.tolist())
                    predicts.append(temp)
                return predicts

            if self.sample:
                probs = F.softmax(beam_scores, dim=-1)
                bests = torch.multinomial(probs, 1).view(-1)
            else:
                bests = beam_scores.argmax(dim=-1)

            for i in range(batch_size):
                best_len = beam_lens[i, bests[i]]
                best_seq = result[i, bests[i], 1:best_len - 1]
                predicts.append(best_seq.tolist())

        return predicts
