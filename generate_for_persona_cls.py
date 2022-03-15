import json
from main_our_v2 import Model as our_model
from main_lost import Model as lost_model
from main_lost_persona import Model as lost_persona_model
from main_transfer_persona import Model as transfer_persona_model
from main_transfer import Model as transfer_model
from main_origin import Model as seq2seq_model

from main_unembedding import Model as unembedding_model
from main_unweight import Model as unweight_model
from main_unpretrain import Model as unpretrain_model
from main_heuristic import Model as heuristic_model


def convert2str_random(dic):
    tag = ''.join(dic['tag'])
    gender = '男性' if dic['gender'] == 'male' else '女性'
    loc = dic['loc']
    res = '性别: ' + gender + ';地域: ' + loc + ';标签: ' + tag
    return res


def convert2str_biased(dic):
    tag = dic['tag']
    gender = '男性' if dic['gender'] == 'male' else '女性'
    loc = dic['loc']
    res = '性别: ' + gender + ';地域: ' + loc + ';标签: ' + tag
    return res


def test(model, input_file, output_file):
    with open(output_file, 'w', encoding='utf8') as fw:
        with open(input_file, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            cnt = 0
            for line in lines[:10]:
                res = {'label': '有'}
                cnt += 1
                line = line.strip('\n')
                data = json.loads(line)
                dialog = data['dialog']
                uid = data['uid']
                if 'responder_profile' in data:
                    responder_profile = data['responder_profile']
                    responder_profile_str = convert2str_biased(responder_profile)
                else:
                    responder_profile = data['response_profile']
                    responder_profile_str = convert2str_random(responder_profile)
                res['responder_profile'] = responder_profile_str
                dialog = [''.join(dialog[i]).replace(' ', '') for i in range(len(dialog))]
                res['dialog'] = dialog
                ans_auto = model.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                res['golden_response'] = ans_auto
                fw.write(json.dumps(res, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # choose = 'unpretrain'
    choose = 'our'
    model = None
    if choose == 'our':
        model = our_model()
    if choose == 'lost':
        model = lost_model()
    if choose == 'transfer':
        model = transfer_model()
    if choose == 'seq2seq':
        model = seq2seq_model()
    if choose == 'lost_persona':
        model = lost_persona_model()
    if choose == 'transfer_persona':
        model = transfer_persona_model()
    if choose == 'unembedding':
        model = unembedding_model()
    if choose == 'unweight':
        model = unweight_model()
    if choose == 'unpretrain':
        model = unpretrain_model()
    if choose == 'heuristic':
        model = heuristic_model()
    files = [['data/test_data_biased.json', 'data/generate/test_data_biased_%s_e27.json' % choose],
             ['data/test_data_random.json', 'data/generate/test_data_random_2k_%s_e27.json' % choose]]
    test(model, files[0][0], files[0][1])
    print('biased ok')
    # test(model, files[1][0], files[1][1])
    print('random ok')
    # test(seq2seq, lost, transfer, our, lost_persona, transfer_persona, files[1][0], files[1][1])
