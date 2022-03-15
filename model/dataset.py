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
from torch.utils.data import Dataset
import json


class SMPDataset(Dataset):
    def __init__(self, paths, vocab, max_lengths=2048):
        if isinstance(paths, str):
            paths = [paths]

        self.vocab = vocab
        self.max_lengths = max_lengths
        self.data = SMPDataset.make_dataset(paths[0], vocab, max_lengths)

    # @staticmethod
    # def make_dataset(paths, vocab, max_lengths):
    #     dataset = []
    #     with open(paths, 'r', encoding='utf8') as fr:
    #         lines = fr.readlines()
    #         for line in lines:
    #             line = line.strip('\n')
    #             data = json.loads(line)
    #             dialog = data['dialog']
    #             profile = data['profile']
    #             uid = data['uid']
    #             if len(dialog) == len(uid) and len(dialog) > 1:
    #                 dialog_ids = [vocab.string2ids(' '.join(i[0].replace(' ', ''))) for i in dialog]
    #                 profile_ids = []
    #                 for i in profile:
    #                     tag = ';'.join(i['tag']).replace(' ', '')
    #                     loc = ';'.join(i['loc'].split()).replace(' ', '')
    #                     gender = '男' if i['gender'] == 'male' else '女'
    #                     persona = '标签:' + tag + ',' + '地点:' + loc + ',' + '性别:' + gender
    #                     profile_ids.append(vocab.string2ids(' '.join(persona)))
    #                 dataset.append([dialog_ids, profile_ids, uid])
    #             else:
    #                 print(data)
    #     return dataset
    #
    # def __len__(self):
    #     return len(self.data) // 20

    def make_dataset(paths, vocab, max_lengths):
        dataset = []
        with open(paths, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            for line in lines:
                line = line.strip('\n')
                # line.replace('\','\)
                # line = line.replace('\\', '\\\\')
                line = line.split('\t')[2].strip('\n').replace('\'', '\"')
                # line = line.replace('\\', '\\\\')
                try:
                    data = json.loads(line)
                except:
                    continue
                dialog = data['dialog']
                profile = data['profile']
                uid = data['uid']
                if len(dialog) == len(uid) and len(dialog) > 1:
                    dialog_ids = [vocab.string2ids(' '.join(i[0].replace(' ', ''))) for i in dialog]
                    profile_ids = []
                    for i in profile:
                        tag = ';'.join(i['tag']).replace(' ', '')
                        loc = ';'.join(i['loc'].split()).replace(' ', '')
                        gender = '男' if i['gender'] == 'male' else '女'
                        persona = '标签:' + tag + ',' + '地点:' + loc + ',' + '性别:' + gender
                        profile_ids.append(vocab.string2ids(' '.join(persona)))
                    dataset.append([dialog_ids, profile_ids, uid])
                else:
                    print(data)
        return dataset

    def __len__(self):
        # return len(self.data) // 20 这个代表的应该是进度条的长度
        return len(self.data)

    def __getitem__(self, idx):
        sample = random.choice(self.data)
        dialog_ids, profile_ids, uid = sample
        l = len(dialog_ids)
        n = random.choice(range(1, l))
        profile = profile_ids[uid[n]]
        history = dialog_ids[:n]
        y = dialog_ids[n]
        profile = [self.vocab.eos_id] + profile + [self.vocab.eos_id]
        history_cat = [self.vocab.eos_id]
        for i in history[:-1]:
            history_cat.extend(i)
            history_cat.extend([self.vocab.spl_id])
        history_cat.extend(history[-1])
        history_cat.extend([self.vocab.eos_id])
        y = [self.vocab.eos_id] + y + [self.vocab.eos_id]
        profile = profile[:64]
        history_cat = history_cat[-256:]
        y = y[:64]
        return profile, history_cat, y


class SMPDataset_soft(Dataset):
    def __init__(self, paths, vocab, max_lengths=2048):
        if isinstance(paths, str):
            paths = [paths]
        self.vocab = vocab
        self.max_lengths = max_lengths
        with open('data/tag2cnt.txt', 'r', encoding='utf8') as fr:
            tags = [line.split('\t')[0] for line in fr.readlines()][:500]
        self.tag2id = {j: i + 2 for i, j in enumerate(tags)}
        self.id2tag = {i + 2: j for i, j in enumerate(tags)}
        with open('data/loc2cnt.txt', 'r', encoding='utf8') as fr:
            locs = [line.split('\t')[0] for line in fr.readlines()][1:]
        self.loc2id = {j: i + 2 for i, j in enumerate(locs)}
        self.id2loc = {i + 2: j for i, j in enumerate(locs)}
        self.gender2id = {'男': 1, '女': 2}
        self.data = self.make_dataset(paths[0], vocab, max_lengths)

    # @staticmethod
    def make_dataset(self, paths, vocab, max_lengths):
        dataset = []
        with open(paths, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            for line in lines[:50000]:
                label, _, line = line.strip('\n').split('\t')
                label = int(label)
                data = json.loads(line)
                dialog = data['dialog']
                profile = data['profile']
                uid = data['uid']
                if len(dialog) == len(uid) and len(dialog) > 1:
                    dialog_ids = [vocab.string2ids(' '.join(i[0].replace(' ', ''))) for i in dialog]
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
                        profile_ids.append(vocab.string2ids(' '.join(persona)))
                    dataset.append([dialog_ids, profile_ids, uid, profile_embedding_ids, label])
                else:
                    print(data)
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = random.choice(self.data)
        sample = self.data[idx]
        dialog_ids, profile_ids, uid, profile_embedding_ids, label = sample
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
        return profile, history_cat, y, gender_cat, loc_cat, tag_cat, label


class SMPDataset_our(Dataset):
    def __init__(self, paths, vocab, max_lengths=2048, is_test=False):
        if isinstance(paths, str):
            paths = [paths]
        self.vocab = vocab
        self.max_lengths = max_lengths
        with open('data/tag2cnt.txt', 'r', encoding='utf8') as fr:
            tags = [line.split('\n')[0] for line in fr.readlines()][:500]
        self.tag2id = {j: i + 2 for i, j in enumerate(tags)}
        self.id2tag = {i + 2: j for i, j in enumerate(tags)}
        with open('data/loc2cnt.txt', 'r', encoding='utf8') as fr:
            locs = [line.split('\n')[0] for line in fr.readlines()][1:]
        self.loc2id = {j: i + 2 for i, j in enumerate(locs)}
        self.id2loc = {i + 2: j for i, j in enumerate(locs)}
        self.gender2id = {'男': 1, '女': 2}
        self.data = self.make_dataset(paths[0], vocab, max_lengths, is_test)

    # @staticmethod
    def make_dataset(self, paths, vocab, max_lengths, is_test):
        dataset = []
        import tqdm
        if is_test:
            cut = 2000
        else:
            cut = 50000
        a = 0
        with open(paths, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            for line in tqdm.tqdm(lines[:cut]):
                flag1 = True
                ##TODO check the meaning
                # line_info = json.loads(line.split('\t')[0].strip('\n').replace('\'', '\"').replace('\\', '\\\\'))
                # line = line_info['dialog']
                _, label, choose, line = line.strip('\n').split('\t')
                label = int(label)
                choose = [int(i) for i in choose.split(';')]
                try:
                    data = json.loads(line.replace('\'', '\"'))
                except:
                    a += 1
                    continue  ##TODO
                # dialog = line_info['dialog']
                # profile = line_info['profile']
                # uid = line_info['uid']
                dialog = data['dialog']
                profile = data['profile']
                uid = data['uid']
                if len(dialog) == len(uid) and len(dialog) > 1:
                    dialog_ids = [vocab.string2ids(' '.join(i[0].replace(' ', ''))) for i in dialog]
                    profile_ids = []
                    profile_embedding_ids = []
                    for i in profile:
                        """
                        persona with age attribute
                        """
                        temp = []
                        tag = ';'.join(i['tag']).replace(' ', '')
                        loc = ';'.join(i['loc'].split()).replace(' ', '')
                        gender = '男' if i['gender'] == 'male' else '女'
                        # temp.append(age_str[str(age)]) ##组成为 gender, age, interests
                        temp.append(self.gender2id[gender])  ##男1 女2
                        loc_list = loc.split(';')
                        if len(loc_list):
                            if loc_list[0] in self.loc2id:
                                temp.append(self.loc2id[loc_list[0]])
                            else:
                                temp.append(1)
                        else:
                            temp.append(1)
                        tag_list = tag.split(';')
                        rank = []
                        if tag_list:
                            for j in tag_list:  ## 选择排名考前的个性标签
                                if j in self.tag2id:
                                    rank.append(self.tag2id[j])
                            try:
                                rank.sort()
                                temp.append(rank[0])
                            except:
                                pass  ## 没有被
                            # temp.append(self.tag2id[j])
                        if len(temp) == 2:
                            temp.append(1)  # 1 for 没有
                        assert (len(temp) == 3)
                        profile_embedding_ids.append(temp)  ##profile_embbedding needs average
                        persona = '性别:' + gender + ',' + '地点:' + loc + ',' + '标签:' + tag  ##还是以字符串的形式直接嵌入
                        profile_ids.append(vocab.string2ids(' '.join(persona)))
                        try:
                            assert (max(choose) <= len(dialog) - 1)
                            assert (max(choose) <= len(uid) - 1)
                            assert (len(dialog) == len(uid))
                        except:
                            flag1 = False
                    if flag1:
                        dataset.append([dialog_ids, profile_ids, uid, profile_embedding_ids, label, choose])
                else:  ##pro_file_id 有两个 profile_embbeding 可能缺失, choose是除了第一句之后的选择??
                    print(data)
        return dataset
        # data = json.loads(line)
        # # dialog = line_info['dialog']
        # # profile = line_info['profile']
        # # uid = line_info['uid']
        # dialog = data['dialog']
        # profile = data['profile']
        # uid = data['uid']
        # if len(dialog) == len(uid) and len(dialog) > 1:
        #     dialog_ids = [vocab.string2ids(' '.join(i[0].replace(' ', ''))) for i in dialog]
        #     profile_ids = []
        #     profile_embedding_ids = []
        #     for i in profile:
        #         temp = []
        #         tag = ';'.join(i['tag']).replace(' ', '')
        #         loc = ';'.join(i['loc'].split()).replace(' ', '')
        #         gender = '男' if i['gender'] == 'male' else '女'
        #         temp.append(self.gender2id[gender])  ##男1 女2
        #         loc_list = loc.split(';')
        #         if len(loc_list):
        #             if loc_list[0] in self.loc2id:
        #                 temp.append(self.loc2id[loc_list[0]])
        #             else:
        #                 temp.append(1)
        #         else:
        #             temp.append(1)
        #         tag_list = tag.split(';')
        #         for j in tag_list:
        #             if j in self.tag2id:
        #                 temp.append(self.tag2id[j])
        #                 break  # 只计算一个兴趣标签吗?
        #         if len(temp) == 2:
        #             temp.append(1)
        #         profile_embedding_ids.append(temp)  ##profile_embbedding needs average
        #         persona = '性别:' + gender + ',' + '地点:' + loc + ',' + '标签:' + tag  ##还是以字符串的形式直接嵌入
        #         profile_ids.append(vocab.string2ids(' '.join(persona)))
        #     dataset.append([dialog_ids, profile_ids, uid, profile_embedding_ids, label, choose])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = random.choice(self.data)
        sample = self.data[idx]
        try:
            dialog_ids, profile_ids, uid, profile_embedding_ids, label, choose = sample
        except:
            print(11)
        l = len(dialog_ids)
        n = random.choice(choose)  ##不管有没有个性突出，从可选的选项中选择  choose = [1, 2] uid = [0, 1] dialog_ids != len(uid)
        try:
            profile = profile_ids[uid[n]]  ## 这里指明了，这是被选出来的n的 profile,即 target的 persona，太偏颇了，这里
        except:
            print(1)
        # profile_embedding = profile_embedding_ids[uid[n]]
        history = dialog_ids[:n]
        y = dialog_ids[n]  ##y 是抽出来的对话
        profile = [self.vocab.eos_id] + profile + [self.vocab.eos_id]  ##target persona
        history_cat = [self.vocab.eos_id]
        gender_cat = [profile_embedding_ids[uid[0]][0]]
        loc_cat = [profile_embedding_ids[uid[0]][1]]  # break 的作用 就是 做成三元组的形式方便这里选出来
        tag_cat = [profile_embedding_ids[uid[0]][2]]  ##恒定第一个发话的人
        for k in range(n):
            temp = history[k] + [self.vocab.spl_id]
            history_cat.extend(temp)
            gender_cat.extend([profile_embedding_ids[uid[k]][0]] * len(temp))
            loc_cat.extend(
                [profile_embedding_ids[uid[k]][1]] * len(temp))  # 这里给定了一个开头然后对于备选的 历史进行编码操作,斗志选了一个特征,然后根据长度来编码
            tag_cat.extend([profile_embedding_ids[uid[k]][2]] * len(temp))  ##就是这里体现了，如何蒋不同的特征的embbeding进行嵌入，准备数据
        history_cat[-1] = self.vocab.eos_id  ## </p> 改成结尾  这里对应的嵌入
        y = [self.vocab.eos_id] + y + [self.vocab.eos_id]
        profile = profile[:48]
        history_cat = history_cat[-128:]
        gender_cat = gender_cat[-128:]
        loc_cat = loc_cat[-128:]
        tag_cat = tag_cat[-128:]
        y = y[:32]
        return profile, history_cat, y, gender_cat, loc_cat, tag_cat, label
        ## return profile（str) + history-(/p + /s) fill with special token +, y是抽出来的对话, (gender_cat, loc_cat, tag_cat tuple) * len(token)
        ##主要看的是history的长度


class SMPDataset_lost(Dataset):
    def __init__(self, paths, vocab, max_lengths=2048, istest=False):
        if isinstance(paths, str):
            paths = [paths]

        self.vocab = vocab
        self.max_lengths = max_lengths
        self.data = SMPDataset_lost.make_dataset(paths[0], vocab, max_lengths, istest=False)

    @staticmethod
    def make_dataset(paths, vocab, max_lengths, istest):
        dataset = []
        if istest:
            cut = 2000
        else:
            cut = 50000
        a = 0
        with open(paths, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            for line in lines[:cut]:
                try:
                    _, label, choose, line = line.strip('\n').split('\t')
                except:
                    line = line.strip('\n').split('\t')
                try:
                    data = json.loads(line.replace('\'', '\"'))
                except:
                    continue
                dialog = data['dialog']
                profile = data['profile']
                uid = data['uid']
                # if len(dialog) == len(uid) and len(dialog) > 1 and label == '0':
                if len(dialog) == len(uid) and len(dialog) > 1:
                    dialog_ids = [vocab.string2ids(' '.join(i[0].replace(' ', ''))) for i in dialog]
                    profile_ids = []
                    for i in profile:
                        tag = ';'.join(i['tag']).replace(' ', '')
                        loc = ';'.join(i['loc'].split()).replace(' ', '')
                        gender = '男' if i['gender'] == 'male' else '女'
                        persona = '性别:' + gender + ',' + '地点:' + loc + ',' + '标签:' + tag
                        profile_ids.append(vocab.string2ids(' '.join(persona)))
                    dataset.append([dialog_ids, profile_ids, uid])
                else:
                    # print(data)
                    pass
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = random.choice(self.data)
        sample = self.data[idx]
        dialog_ids, profile_ids, uid = sample
        l = len(dialog_ids)
        n = random.choice(range(1, l))
        profile = profile_ids[uid[n]]
        history = dialog_ids[:n]
        y = dialog_ids[n]
        profile = [self.vocab.eos_id] + profile + [self.vocab.eos_id]
        history_cat = [self.vocab.eos_id]
        for i in history[:-1]:
            history_cat.extend(i)
            history_cat.extend([self.vocab.spl_id])
        history_cat.extend(history[-1])
        history_cat.extend([self.vocab.eos_id])
        y = [self.vocab.eos_id] + y + [self.vocab.eos_id]
        profile = profile[:48]
        history_cat = history_cat[-128:]
        y = y[:32]
        return profile, history_cat, y


class SMPDataset_transfer(Dataset):
    def __init__(self, paths, vocab, max_lengths=2048):
        if isinstance(paths, str):
            paths = [paths]

        self.vocab = vocab
        self.max_lengths = max_lengths
        self.data = SMPDataset_transfer.make_dataset(paths[0], vocab, max_lengths)

    @staticmethod
    def make_dataset(paths, vocab, max_lengths):
        dataset = []
        with open(paths, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            for line in lines:
                label, choose, line = line.strip('\n').split('\t')
                data = json.loads(line)
                dialog = data['dialog']
                profile = data01

                ['profile']
                uid = data['uid']
                if len(dialog) == len(uid) and len(dialog) > 1 and label == '0':
                    dialog_ids = [vocab.string2ids(' '.join(i[0].replace(' ', ''))) for i in dialog]
                    profile_ids = []
                    for i in profile:
                        tag = ';'.join(i['tag']).replace(' ', '')
                        loc = ';'.join(i['loc'].split()).replace(' ', '')
                        gender = '男' if i['gender'] == 'male' else '女'
                        persona = '性别:' + gender + ',' + '地点:' + loc + ',' + '标签:' + tag
                        profile_ids.append(vocab.string2ids(' '.join(persona)))
                    dataset.append([dialog_ids, profile_ids, uid])
                else:
                    # print(data)
                    pass
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = random.choice(self.data)
        sample = self.data[idx]
        dialog_ids, profile_ids, uid = sample
        l = len(dialog_ids)
        n = random.choice(range(1, l))
        source = [self.vocab.eos_id]
        target = [self.vocab.pad_id]
        id_user = [self.vocab.eos_id]
        profile = profile_ids[uid[n]][:48]
        id_profile = self.vocab.p2_id if uid[n] else self.vocab.p1_id
        source.extend(profile + [self.vocab.eos_id])
        target.extend([self.vocab.pad_id] * (len(profile) + 1))
        id_user.extend([id_profile] * (len(profile) + 1))
        # process history
        temp_s = []
        temp_t = []
        temp_u = []
        for i in range(n):
            temp_s.extend([self.vocab.spl_id] + dialog_ids[i])
            temp_t.extend([self.vocab.pad_id] * (len(dialog_ids[i]) + 1))
            if i % 2 == 0:
                temp_u.extend([self.vocab.p1_id] * (len(dialog_ids[i]) + 1))
            else:
                temp_u.extend([self.vocab.p2_id] * (len(dialog_ids[i]) + 1))
        source.extend(temp_s[-128:])
        target.extend(temp_t[-128:])
        id_user.extend(temp_u[-128:])
        # process target
        source.extend([self.vocab.spl_id] + dialog_ids[n] + [self.vocab.eos_id])
        target.extend([self.vocab.spl_id] + dialog_ids[n] + [self.vocab.eos_id])
        id_user.extend([id_profile] * (len(dialog_ids[n]) + 1) + [self.vocab.eos_id])
        source = source[:256]
        target = target[:256]
        id_user = id_user[:256]
        return source, target, id_user


##TODO 2020/7/1
class SMPDataset_v3(Dataset):
    def __init__(self, paths, vocab, max_lengths=2048, is_test=False):
        if isinstance(paths, str):
            paths = [paths]
        self.vocab = vocab
        self.max_lengths = max_lengths
        with open('data/data_v2/tag2cnt.txt', 'r', encoding='utf8') as fr:
            tags = [line.split('\t')[0] for line in fr.readlines()][:500]
        self.tag2id = {j.split()[0]: i + 1 for i, j in enumerate(tags)}
        self.id2tag = {i + 1: j.split()[0] for i, j in enumerate(tags)}
        with open('data/data_v2/loc2cnt.txt', 'r', encoding='utf8') as fr:
            locs = [line.split('\t')[0] for line in fr.readlines()][1:255]  ##255 - 1 + 1
        self.loc2id = {j.split()[0]: i + 1 for i, j in enumerate(locs)}  ## 跳过了其他
        self.id2loc = {i + 1: j.split()[0] for i, j in enumerate(locs)}
        self.gender2id = {'男': 1, '女': 2}
        self.data = self.make_dataset(paths[0], vocab, max_lengths, is_test)

    # @staticmethod
    def make_dataset(self, paths, vocab, max_lengths, is_test):
        dataset = []
        import tqdm
        if is_test:
            cut = 2000
        else:
            cut = 50000
        a = 0
        with open(paths, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            for line in tqdm.tqdm(lines[:cut]):
                ##TODO check the meaning
                # line_info = json.loads(line.split('\t')[0].strip('\n').replace('\'', '\"').replace('\\', '\\\\'))
                # line = line_info['dialog']
                # label, choose, line = line.strip('\n').split('\t')
                # bilabel, label, choose, line = line.split('\t')
                flag1 = True
                label, _, choose, line = line.split('\t')
                # label = int(label)
                label = [eval(i) for i in label.split(',')]
                choose = [int(i) for i in choose.split(';')]
                try:
                    data = json.loads(line.replace('\'', '\"'))
                except:
                    a += 1
                    continue
                # dialog = line_info['dialog']
                # profile = line_info['profile']
                # uid = line_info['uid']
                dialog = data['dialog']
                profile = data['profile']
                uid = data['uid']
                if len(dialog) == len(uid) and len(dialog) > 1:
                    dialog_ids = [vocab.string2ids(' '.join(i[0].replace(' ', ''))) for i in dialog]
                    profile_ids = []
                    profile_embedding_ids = []
                    for i in profile:
                        """
                        persona with age attribute
                        """
                        temp = []
                        tag = ';'.join(i['tag']).replace(' ', '')
                        loc = ';'.join(i['loc'].split()).replace(' ', '')
                        gender = '男' if i['gender'] == 'male' else '女'
                        # temp.append(age_str[str(age)]) ##组成为 gender, age, interests
                        temp.append(self.gender2id[gender])  ##男1 女2
                        loc_list = loc.split(';')
                        if len(loc_list):
                            if loc_list[0] in self.loc2id:
                                temp.append(self.loc2id[loc_list[0]])
                            else:
                                temp.append(1)
                        else:
                            temp.append(1)
                        tag_list = tag.split(';')
                        rank = []
                        if tag_list:
                            for j in tag_list:  ## 选择排名考前的个性标签
                                if j in self.tag2id:
                                    rank.append(self.tag2id[j])
                            try:
                                rank.sort()
                                temp.append(rank[0])
                            except:
                                pass  ## 没有被
                            # temp.append(self.tag2id[j])
                        if len(temp) == 2:
                            temp.append(1)  # 1 for 没有
                        assert (len(temp) == 3)
                        profile_embedding_ids.append(temp)  ##profile_embbedding needs average
                        persona = '性别:' + gender + ',' + '地点:' + loc + ',' + '标签:' + tag  ##还是以字符串的形式直接嵌入
                        profile_ids.append(vocab.string2ids(' '.join(persona)))
                        try:
                            assert (max(choose) <= len(dialog) - 1)
                            assert (max(choose) <= len(uid) - 1)
                            assert (len(dialog) == len(uid))
                        except:
                            flag1 = False
                    if flag1:
                        dataset.append([dialog_ids, profile_ids, uid, profile_embedding_ids, label, choose])
                else:  ##pro_file_id 有两个 profile_embbeding 可能缺失, choose是除了第一句之后的选择??
                    print(data)  # 三元组：组成为 gender, age, interests
        print(a)  ##一共9个不符合? #7836个
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = random.choice(self.data)
        sample = self.data[idx]
        dialog_ids, profile_ids, uid, profile_embedding_ids, label, choose = sample
        l = len(dialog_ids)
        n = random.choice(choose)  ##不管有没有个性突出，从可选的选项中选择
        if n % 2 == 0:
            a = 1
            b = 0
        else:
            a = 0
            b = 1
        target_profile = profile_ids[uid[a]]  ## 这里指明了，这是被选出来的n的 profile,即 target的 persona，太偏颇了，这里
        # del profile_ids[uid[n]]  ## anothor profile
        source_profile = profile_ids[uid[b]]  ## 这里指明了，这是被选出来的n的 profile,即 target的 persona，太偏颇了，这里

        # profile_embedding = profile_embedding_ids[uid[n]]  # 这里指明的是Target_embbedding

        history = dialog_ids[:n]
        response = dialog_ids[n]  ##y 是抽出来的对话

        target_profile = [self.vocab.eos_id] + target_profile + [self.vocab.eos_id]  ##target persona
        source_profile = [self.vocab.eos_id] + source_profile + [self.vocab.eos_id]  ##source persona

        history_cat = [self.vocab.eos_id]
        gender_cat = [profile_embedding_ids[uid[0]][0]]
        loc_cat = [profile_embedding_ids[uid[0]][1]]  # break 的作用 就是 做成三元组的形式方便这里选出来
        tag_cat = [profile_embedding_ids[uid[0]][2]]  ##恒定第一个发话的人
        for k in range(n):
            temp = history[k] + [self.vocab.spl_id]
            history_cat.extend(temp)
            gender_cat.extend([profile_embedding_ids[uid[k]][0]] * len(temp))
            loc_cat.extend(  ###CONTEXT dialogue embbedding
                [profile_embedding_ids[uid[k]][1]] * len(temp))  # 这里给定了一个开头然后对于备选的 历史进行编码操作,斗志选了一个特征,然后根据长度来编码
            tag_cat.extend([profile_embedding_ids[uid[k]][2]] * len(temp))  ##就是这里体现了，如何蒋不同的特征的embbeding进行嵌入，准备数据
        history_cat[-1] = self.vocab.eos_id  ## </p> 改成结尾  这里对应的嵌入
        response = [self.vocab.eos_id] + response + [self.vocab.eos_id]

        source_profile = source_profile[:48]
        target_profile = target_profile[:48]

        history_cat = history_cat[-128:]

        gender_cat = gender_cat[-128:]
        loc_cat = loc_cat[-128:]
        tag_cat = tag_cat[-128:]
        response = response[:32]

        label = label[n]

        return source_profile, target_profile, history_cat, response, gender_cat, loc_cat, tag_cat, label
        ## return profile（str) + history-(/p + /s) fill with special token +, y是抽出来的对话, (gender_cat, loc_cat, tag_cat tuple) * len(token)
        ##主要看的是history的长度
