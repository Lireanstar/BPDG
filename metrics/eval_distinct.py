#!/usr/bin/env python3
# coding:utf-8

# Copyright (c) Tsinghua university conversational AI group (THU-coai).
# This source code is licensed under the MIT license.
"""Script for the Evaluation of Chinese Human-Computer Dialogue Technology (SMP2019-ECDT) Task2.
This script evaluates the distinct[1] of the submitted model.
This uses a the version of the dataset which does not contain the "Golden Response" .
Leaderboard scores will be run in the same form but on a hidden test set.

reference:

[1] Li, Jiwei, et al. "A diversity-promoting objective function for neural conversation models."
    arXiv preprint arXiv:1510.03055 (2015).

This requires each team to implement the following function:
def gen_response(self, contexts):
    return a list of responses for each context
    Arguments:
    contexts -- a list of context, each context contains dialogue histories and personal profiles of every speaker
    Returns a list, where each element is the response of the corresponding context
"""
import json
import sys
import codecs
sys.path.append("./")
#from main_our_v2 import Model
from main_our import Model
#from main_lost import Model
#from main_transfer import Model
#from main_lost_persona import Model
#from main_transfer_persona import Model
#from main_unweight import Model
#from main_unembedding import Model
# from main_unpretrain import Model
#from main_origin import Model
#from main_heuristic import Model

def read_dialog(file):
    """
    Read dialogs from file
    :param file: str, file path to the dataset
    :return: list, a list of dialogue (context) contained in file
    """
    with codecs.open(file, 'r', 'utf-8') as f:
        contents = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    return [json.loads(i) for i in contents]


def count_ngram(hyps_resp, n):
    """
    Count the number of unique n-grams
    :param hyps_resp: list, a list of responses
    :param n: int, n-gram
    :return: the number of unique n-grams in hyps_resp
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(' '.join(resp[i: i + n]))
    return len(ngram)


def eval_distinct(hyps_resp):
    """
    compute distinct score for the hyps_resp
    :param hyps_resp: list, a list of hyps responses
    :return: average distinct score for 1, 2-gram
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    hyps_resp = [(' '.join(i)).split() for i in hyps_resp]
    num_tokens = sum([len(i) for i in hyps_resp])
    dist1 = count_ngram(hyps_resp, 1) / float(num_tokens)
    dist2 = count_ngram(hyps_resp, 2) / float(num_tokens)
    print(dist1, dist2)
    return (dist1 + dist2) / 2.0


if __name__ == '__main__':
    model = Model()

    if len(sys.argv) < 3:
        print('Too few args for this script')

    # random_test = sys.argv[1]
    # biased_test = sys.argv[2]

    random_test = '/home/zhengyinhe/DialogueFilter/data/persona_compe/test/test_biased.json'
    biased_test = '/home/zhengyinhe/DialogueFilter/data/persona_compe/test/test_biased.json'
    random_test_data = read_dialog(random_test)
    biased_test_data = read_dialog(biased_test)

    for i in random_test_data:
        if 'golden_response' in i:
            del i['golden_response']
    for i in biased_test_data:
        if 'golden_response' in i:
            del i['golden_response']

    # random_hyps_resp = model.gen_response(random_test_data)
    # biased_hyps_resp = model.gen_response(biased_test_data)
    random_hyps_resp = []
    for count, i in enumerate(random_test_data):
        if count % 100 == 0:
            print(count)
        random_hyps_resp += model.gen_response([i])
        if count % 100 == 0:
            print(' '.join(random_hyps_resp[-1]))
    random_distinct = eval_distinct(random_hyps_resp)
    print('random distinct', random_distinct)

    biased_hyps_resp = []
    for count, i in enumerate(biased_test_data):
        if count % 100 == 0:
            print(count)
        biased_hyps_resp += model.gen_response([i])
        if count % 100 == 0:
            print(' '.join(biased_hyps_resp[-1]))
    biased_distinct = eval_distinct(biased_hyps_resp)
    print('biased distinct', biased_distinct)
    print((random_distinct + biased_distinct) / 2.0)
