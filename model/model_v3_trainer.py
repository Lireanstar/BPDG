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

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import pad_sequence
from .optim import Adam, NoamOpt
from .loss import LabelSmoothingLoss
import json
import logging

logger = logging.getLogger('s2s-our-v2')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('s2s-our-v2.log', encoding='utf-8')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
# 记录一条日志
logger.info('python logging test')


class TrainerSoft:
    def __init__(self, model, train_dataset, test_dataset=None, batch_size=8,
                 batch_split=1, lm_weight=0.5, risk_weight=0, lr=6.25e-5, lr_warmup=2000,
                 n_jobs=0, clip_grad=None, label_smoothing=0, device=torch.device('cuda'),
                 ignore_idxs=[], distributed=False):
        self.model = model.to(device)
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx).to(device)
        self.criterion = LabelSmoothingLoss(n_labels=self.model.n_embeddings, smoothing=label_smoothing,
                                            ignore_index=self.model.padding_idx).to(device)
        self.cls_criterion = nn.CrossEntropyLoss().to(device)  ##new classfication of persona
        # base_optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        # self.optimizer = NoamOpt(self.model.embeddings_size, 0.1, lr_warmup, base_optimizer)

        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        # self.optimizer = NoamOpt(self.model.embeddings_size, 0.1, lr_wwarmup, base_optimizer)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.99,
                                                                    patience=10,
                                                                    verbose=True, min_lr=1e-8)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if distributed else None
        self.train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size // batch_split,
                                           shuffle=(not distributed), num_workers=n_jobs,
                                           collate_fn=self.collate_func)
        if test_dataset is not None:
            self.test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size // batch_split,
                                              shuffle=False,
                                              num_workers=n_jobs, collate_fn=self.collate_func)

        self.batch_split = batch_split
        self.lm_weight = lm_weight
        self.risk_weight = risk_weight
        self.clip_grad = clip_grad
        self.device = device
        self.ignore_idxs = ignore_idxs

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=True)
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def collate_func(self, data):  # dialog_ids, profile_ids, uid, profile_embedding_ids, label, choose
        source_persona_info, target_persona_info, h, y, gender_cat, age_cat, tag_cat, label = zip(*data)
        ## NOTE
        ## source_profile, target_profile, history_cat, response, gender_cat, age_cat, tag_cat, label
        ## persona_info : str(persona), h : history : 0 = <p>, y: target sentence : ground truth
        ## source bounds with the history, target responds to the response
        source_contexts = []
        target_contexts = []

        if max(map(len, source_persona_info)) > 0:  # for source --- history part
            source_persona_info = [torch.tensor(d, dtype=torch.long) for d in source_persona_info]
            source_persona_info = pad_sequence(source_persona_info, batch_first=True,
                                               padding_value=self.model.padding_idx)  # max padding batch_behave
            source_contexts.append(source_persona_info)

        if max(map(len, target_persona_info)) > 0:  # for target --- response part
            target_persona_info = [torch.tensor(d, dtype=torch.long) for d in target_persona_info]
            target_persona_info = pad_sequence(target_persona_info, batch_first=True,
                                               padding_value=self.model.padding_idx)  # max padding
            target_contexts.append(target_persona_info)

        if max(map(len, h)) > 0:
            h = [torch.tensor(d, dtype=torch.long) for d in h]
            h = pad_sequence(h, batch_first=True, padding_value=self.model.padding_idx)
            source_contexts.append(h)
            target_contexts.append(h)
            ##for both the source_contexts and the target context

        ## so the contexts contains two parts 1. the persona string , 2. the history sentences padding part
        gender_cat = [torch.tensor(d, dtype=torch.long) for d in gender_cat]
        gender_cat = pad_sequence(gender_cat, batch_first=True, padding_value=0)

        age_cat = [torch.tensor(d, dtype=torch.long) for d in age_cat]
        age_cat = pad_sequence(age_cat, batch_first=True, padding_value=0)

        tag_cat = [torch.tensor(d, dtype=torch.long) for d in tag_cat]
        tag_cat = pad_sequence(tag_cat, batch_first=True, padding_value=0)

        ## pad the other parts
        y = [torch.tensor(d, dtype=torch.long) for d in y]
        y = pad_sequence(y, batch_first=True, padding_value=self.model.padding_idx)

        label = torch.LongTensor(label)  ##cls targe
        ##TODO for the bi-persona cls????? todo!

        return source_contexts, target_contexts, y, gender_cat, age_cat, tag_cat, label

    def _eval_train(self, epoch, risk_func=None):
        self.model.train()

        tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        loss = 0
        lm_loss = 0
        cls_loss = 0
        # for i, (contexts, targets, gender, age, tag, label) in enumerate(tqdm_data): ## former parts
        for i, (sc_contexts, tg_contexts, response, gender, loc, tag, label) in enumerate(tqdm_data):

            source_contexts, target_contexts, targets = \
                [c.to(self.device) for c in sc_contexts], \
                [c.to(self.device) for c in tg_contexts], \
                response.to(self.device)

            gender = gender.to(self.device)
            loc = loc.to(self.device)
            tag = tag.to(self.device)
            label = label.to(self.device)
            enc_contexts = []

            # lm loss
            batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)

            enc_context_source_persona = self.model.encode(source_contexts[0].clone())
            enc_contexts.append(enc_context_source_persona)

            enc_context_target_persona = self.model.encode(target_contexts[0].clone())  ## for former part
            enc_contexts.append(enc_context_target_persona)

            enc_context_history = self.model.encode(target_contexts[1].clone(), gender=gender, loc=loc,
                                                    tag=tag)
            enc_contexts.append(enc_context_history)
            ##enc_contexts contains 3 th lists , contains the enc_context_source, enc_context_tartget, enc_context_source
            # ## the same
            if self.lm_weight > 0:  ##lm_head : 针对histroy 而言
                context_outputs = self.model.generate(
                    enc_context_history[0])  ##history -> pre_softmax hidden which means the lm
                ignore_mask = torch.stack([target_contexts[1] == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                target_contexts[1].masked_fill_(ignore_mask, self.model.padding_idx)
                prevs, nexts = context_outputs[:, :-1, :].contiguous(), target_contexts[1][:,  ## shifted right
                                                                        1:].contiguous()  ##上下文的logits, 与上下文的ground truth
                batch_lm_loss += self.lm_criterion(prevs.view(-1, prevs.shape[-1]),
                                                   nexts.view(-1))  ##这里的context已经嵌入了 个性化的信息了

            # cls loss
            cls_output = self.model.classify(enc_contexts[2])  ##这个分类关乎到是否persona
            weight = F.softmax(cls_output, dim=-1)
            batch_cls_loss = self.cls_criterion(cls_output,
                                                label)  ##TODO 这里的label 还可以换成 三个部分，oontext, persona1, persona2
            ##TODO , 现在persona
            # s2s loss
            prevs, nexts = targets[:, :-1].contiguous(), targets[:,
                                                         1:].contiguous()  # shifted-right loss means the output+ y (y)
            outputs = self.model.decode(prevs, enc_contexts,
                                        weight)  ##enc_context_history contains persona + history(P tag)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            # optimization
            full_loss = (batch_lm_loss * self.lm_weight + batch_loss + 0.5 * batch_cls_loss) / self.batch_split
            full_loss.backward()
            torch.cuda.empty_cache()

            if (i + 1) % self.batch_split == 0:
                if self.clip_grad is not None:
                    for group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

                self.optimizer.step()
                self.scheduler.step(full_loss)
                self.optimizer.zero_grad()

            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
            loss = (i * loss + batch_loss.item()) / (i + 1)
            cls_loss = (i * cls_loss + batch_cls_loss.item()) / (i + 1)
            # tqdm_data.set_postfix(
            #     {'lm_loss': lm_loss, 'loss': loss, 'cls_loss': cls_loss, 'loss_step': batch_loss.item(),
            #      'batch_cls_loss': batch_cls_loss.item(), 'lr': self.optimizer.rate(),
            #      'step': self.optimizer._step})
            tqdm_data.set_postfix(
                {'lm_loss': lm_loss, 'loss': loss, 'cls_loss': cls_loss, 'loss_step': batch_loss.item(),
                 'batch_cls_loss': batch_cls_loss.item(), 'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
                 'step': self.optimizer._step})

        # log_dict = {'epoch': epoch, 'lm_loss': lm_loss, 'loss': loss, 'cls_loss': cls_loss,
        #             'lr': self.optimizer.rate(), 'step': self.optimizer._step}
        log_dict = {'epoch': epoch, 'lm_loss': lm_loss, 'loss': loss, 'cls_loss': cls_loss,
                    'lr': self.optimizer.state_dict()['param_groups'][0]['lr'], 'step': self.optimizer._step}

        log_dict_json = json.dumps(log_dict, ensure_ascii=False)
        logger.info(log_dict_json)

    def _eval_test(self, metric_funcs={}):

        self.model.eval()
        with torch.no_grad():
            tqdm_data = tqdm(self.test_dataloader, desc='Test')
            loss = 0
            lm_loss = 0
            cls_loss = 0
            metrics = {name: 0 for name in metric_funcs.keys()}
            for i, (sc_contexts, tg_contexts, response, gender, loc, tag, label) in enumerate(tqdm_data):
                source_contexts, target_contexts, targets = \
                    [c.to(self.device) for c in sc_contexts], \
                    [c.to(self.device) for c in tg_contexts], \
                    response.to(self.device)
                gender = gender.to(self.device)
                loc = loc.to(self.device)
                tag = tag.to(self.device)
                label = label.to(self.device)
                enc_contexts = []
                # lm loss
                batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)

                enc_context_source_persona = self.model.encode(source_contexts[0].clone())
                enc_contexts.append(enc_context_source_persona)

                enc_context_target_persona = self.model.encode(target_contexts[0].clone())  ## for former part
                enc_contexts.append(enc_context_target_persona)

                enc_context_history = self.model.encode(target_contexts[1].clone(), gender=gender, loc=loc,
                                                        tag=tag)
                enc_contexts.append(enc_context_history)

                if self.lm_weight > 0:  ##lm_head : 针对histroy 而言
                    context_outputs = self.model.generate(
                        enc_context_history[0])  ##history -> pre_softmax hidden which means the lm
                    ignore_mask = torch.stack([target_contexts[1] == idx for idx in self.ignore_idxs], dim=-1).any(
                        dim=-1)
                    target_contexts[1].masked_fill_(ignore_mask, self.model.padding_idx)
                    prevs, nexts = context_outputs[:, :-1, :].contiguous(), target_contexts[1][:,  ## shifted right
                                                                            1:].contiguous()  ##上下文的logits, 与上下文的ground truth
                    batch_lm_loss += self.lm_criterion(prevs.view(-1, prevs.shape[-1]),
                                                       nexts.view(-1))  ##这里的context已经嵌入了 个性化的信息了

                # cls loss
                cls_output = self.model.classify(enc_contexts[2])  ##这个分类关乎到是否persona
                weight = F.softmax(cls_output, dim=-1)
                batch_cls_loss = self.cls_criterion(cls_output,
                                                    label)  ##TODO 这里的label 还可以换成 三个部分，oontext, persona1, persona2
                ##TODO , 现在persona
                # s2s loss
                prevs, nexts = targets[:, :-1].contiguous(), targets[:,
                                                             1:].contiguous()  # shifted-right loss means the output+ y (y)
                outputs = self.model.decode(prevs, enc_contexts,
                                            weight)  ##enc_context_history contains persona + history(P tag)
                outputs = F.log_softmax(outputs, dim=-1)
                batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

                # predictions = self.model.beam_search(enc_contexts, weight=weight)
                predictions = self.model.beam_search(enc_contexts, weight=weight)
                target_lens = targets.ne(self.model.padding_idx).sum(dim=-1)
                targets = [t[1:l - 1].tolist() for t, l in zip(targets, target_lens)]

                lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
                loss = (i * loss + batch_loss.item()) / (i + 1)
                cls_loss = (i * cls_loss + batch_cls_loss.item()) / (i + 1)

                for name, func in metric_funcs.items():
                    score = func(predictions, targets)
                    metrics[name] = (metrics[name] * i + score) / (i + 1)

                tqdm_data.set_postfix(dict({'lm_loss': lm_loss, 'loss': loss, 'cls_loss': cls_loss}, **metrics))
                torch.cuda.empty_cache()
        log_dict = dict({'lm_loss': lm_loss, 'loss': loss, 'cls_loss': cls_loss}, **metrics)
        log_dict_json = json.dumps(log_dict, ensure_ascii=False)
        logger.info(log_dict_json)

    def test(self, metric_funcs={}):
        if hasattr(self, 'test_dataloader'):
            self._eval_test(metric_funcs)

    def train(self, start_epoch, epochs, after_epoch_funcs=[], risk_func=None):
        for epoch in range(start_epoch, epochs):
            self._eval_train(epoch, risk_func)
            torch.cuda.empty_cache()
            if epoch % 2 == 0 and epoch > 0:
                for func in after_epoch_funcs:
                    func(epoch)
            # if epoch % 2 == 0 and epoch > 0:
            # for func in after_epoch_funcs:
            #     func(epoch)
