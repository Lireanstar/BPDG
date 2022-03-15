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
logger = logging.getLogger('lm-transfer')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('lm-transfer.log', encoding='utf-8')
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


class TrainerTransfer:
    def __init__(self, model, train_dataset, test_dataset=None, batch_size=8,
                 batch_split=1, lm_weight=0.5, risk_weight=0, lr=6.25e-5, lr_warmup=2000,
                 n_jobs=0, clip_grad=None, label_smoothing=0, device=torch.device('cuda'),
                 ignore_idxs=[], distributed=False):
        self.model = model.to(device)
        # self.lm_criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx).to(device)
        self.criterion = LabelSmoothingLoss(n_labels=self.model.n_embeddings, smoothing=label_smoothing,
                                            ignore_index=self.model.padding_idx).to(device)
        base_optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.optimizer = NoamOpt(self.model.embeddings_size, 0.1, lr_warmup, base_optimizer)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset) if distributed else None

        self.train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size // batch_split,
                                           shuffle=(not distributed), num_workers=n_jobs,
                                           collate_fn=self.collate_func_lm)
        if test_dataset is not None:
            self.test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size // batch_split,
                                              shuffle=False,
                                              num_workers=n_jobs, collate_fn=self.collate_func_lm)

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
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def collate_func_lm(self, data):
        # print(data)
        # y = zip(*data)
        source, target, id_user = zip(*data)
        source = [torch.tensor(d, dtype=torch.long) for d in source]
        source = pad_sequence(source, batch_first=True, padding_value=self.model.padding_idx)
        target = [torch.tensor(d, dtype=torch.long) for d in target]
        target = pad_sequence(target, batch_first=True, padding_value=self.model.padding_idx)
        id_user = [torch.tensor(d, dtype=torch.long) for d in id_user]
        id_user = pad_sequence(id_user, batch_first=True, padding_value=self.model.padding_idx)
        return source, target, id_user

    def _eval_train(self, epoch, risk_func=None):
        self.model.train()

        tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        loss = 0
        for i, (source, target, id_user) in enumerate(tqdm_data):
            source = source.to(self.device)
            target = target.to(self.device)
            id_user = id_user.to(self.device)
            # print(targets, targets.size())
            # s2s loss
            prevs, nexts = source[:, :-1].contiguous(), target[:, 1:].contiguous()
            id = id_user[:, :-1].contiguous()
            outputs = self.model(prevs, id)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
            # optimization
            full_loss = batch_loss / self.batch_split
            full_loss.backward()

            if (i + 1) % self.batch_split == 0:
                if self.clip_grad is not None:
                    for group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

                self.optimizer.step()
                # self.scheduler.step()
                self.optimizer.zero_grad()

            loss = (i * loss + batch_loss.item()) / (i + 1)
            tqdm_data.set_postfix({'loss': loss, 'loss_step': batch_loss.item()})
        log_dict = {'epoch': epoch, 'loss': loss}
        log_dict_json = json.dumps(log_dict, ensure_ascii=False)
        logger.info(log_dict_json)

    def _eval_test(self, metric_funcs={}):
        self.model.eval()
        tqdm_data = tqdm(self.test_dataloader, desc='Test')
        loss = 0
        for i, (source, target, id_user) in enumerate(tqdm_data):
            source = source.to(self.device)
            target = target.to(self.device)
            id_user = id_user.to(self.device)
            # print(targets, targets.size())
            # s2s loss
            prevs, nexts = source[:, :-1].contiguous(), target[:, 1:].contiguous()
            id = id_user[:, :-1].contiguous()
            outputs = self.model(prevs, id)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))
            loss = (i * loss + batch_loss.item()) / (i + 1)
            tqdm_data.set_postfix({'eval_loss': loss, 'loss_step': batch_loss.item()})
        log_dict = {'eval_loss': loss}
        log_dict_json = json.dumps(log_dict, ensure_ascii=False)
        logger.info(log_dict_json)

    def test(self, metric_funcs={}):
        if hasattr(self, 'test_dataloader'):
            self._eval_test(metric_funcs)

    def train(self, start_epoch, epochs, after_epoch_funcs=[], risk_func=None):
        print(start_epoch, epochs)
        for epoch in range(start_epoch, epochs):
            self._eval_train(epoch, risk_func)
            # if epoch % 10 == 0 and epoch > 0:
            for func in after_epoch_funcs:
                func(epoch)
