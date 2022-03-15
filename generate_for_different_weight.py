import json
import torch
from main_our_v2 import Model as our_model

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

def test(model, input_file, output_file, weight):
    with open(output_file, 'w', encoding='utf8') as fw:
        with open(input_file, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            cnt = 0
            for line in lines:
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
                ans_auto = model.gen_response([data], weight_i=weight)
                ans_auto = ''.join(ans_auto[0])
                res['golden_response'] = ans_auto
                fw.write(json.dumps(res, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    model = our_model()
    weights = [torch.Tensor([[0, 1]]), torch.Tensor([[0.2, 0.8]]), torch.Tensor([[0.4, 0.6]]),
               torch.Tensor([[0.6, 0.4]]), torch.Tensor([[0.8, 0.2]]), torch.Tensor([[1, 0]])]
    weights = [i.to('cuda') for i in weights]
    # for i in range(len(weights)):
    i = 5
    weight = weights[i]
    files = [['data/test_data_biased.json', 'data/generate/test_data_biased_our_v2e39_weight%s.json'%str(2*i)],
             ['data/test_data_random.json', 'data/generate/test_data_random_2k_our_v2e39_weight%s.json'%str(2*i)]]
    test(model, files[0][0], files[0][1], weight)
    print('biased ok')
    test(model, files[1][0], files[1][1], weight)
    print('random ok')
        # test(seq2seq, lost, transfer, our, lost_persona, transfer_persona, files[1][0], files[1][1])