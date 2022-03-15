import json
import torch
from main_our_v2 import Model as our_model

def convert2str_random(dic):
    tag = ''.join(dic['tag'])
    gender = '男' if dic['gender'] == 'male' else '女'
    loc = ';'.join(dic['loc'].split())
    res = '性别: ' + gender + '，地域: ' + loc + '，标签: ' + tag
    return res

def convert2str_biased(dic):
    tag = dic['tag']
    gender = '男' if dic['gender'] == 'male' else '女'
    loc = ';'.join(dic['loc'].split())
    res = '性别: ' + gender + '，地域: ' + loc + '，标签: ' + tag
    return res

def test(our, input_file, output_file):
    with open(output_file, 'w', encoding='utf8') as fw:
        with open(input_file, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            weights = [torch.Tensor([[0, 1]]), torch.Tensor([[0.2, 0.8]]), torch.Tensor([[0.4, 0.6]]),
                       torch.Tensor([[0.6, 0.4]]), torch.Tensor([[0.8, 0.2]]), torch.Tensor([[1, 0]])]
            weights = [i.to('cuda') for i in weights]
            cnt = 0
            for line in lines[:10]:
                cnt += 1
                line = line.strip('\n')
                data = json.loads(line)
                dialog = data['dialog']
                uid = data['uid']
                profile_all = data['profile']
                if 'responder_profile' in data:
                    responder_profile = data['responder_profile']
                    responder_profile_str = convert2str_biased(responder_profile)
                    profile_all = ['persona ' +  str(i) + '\t' +  convert2str_biased(profile_all[i]) for i in range(len(profile_all))]
                else:
                    responder_profile = data['response_profile']
                    responder_profile_str = convert2str_random(responder_profile)
                    profile_all = ['persona ' + str(i) + '\t' + convert2str_random(profile_all[i]) for i in range(len(profile_all))]
                golden_response = data['golden_response']
                golden_response_str = ''.join(golden_response).replace(' ', '')
                dialog = ['P' + str(i%2) + '\t' + ''.join(dialog[i]).replace(' ', '') for i in range(len(dialog))]
                # profile_all_str = '\n\t'.join([json.dumps(i, ensure_ascii=False) for i in profile_all])
                # responder_profile_str = json.dumps(responder_profile, ensure_ascii=False)
                fw.write('dialogue ' + str(cnt) + '\n')
                fw.write('all personas: ' + '\n')
                for i in profile_all:
                    fw.write(i + '\n')
                fw.write('history: ' + '\n')
                for i in dialog:
                    fw.write(i + '\n')
                fw.write('responser persona' + '\t' + responder_profile_str + '\n')
                fw.write('responses: ' + '\n')
                fw.write('golden_response' + '\t' + golden_response_str + '\n')
                # our model predict
                ans_auto = our.gen_response([data], human=True)
                ans_auto = ''.join(ans_auto[0])
                fw.write('our with auto' + '\t' + ans_auto + '\n')
                for i in range(len(weights)):
                    ans_persona = our.gen_response([data], weight_i=weights[i], human=True)
                    ans_persona = ''.join(ans_persona[0])
                    fw.write('persona weight ' + str(0.2*i) + ': ' + '\t' + ans_persona + '\n')
                fw.write('\n\n\n')

if __name__ == '__main__':
    our = our_model()
    # files = [['data/test_data_biased200.json', 'data/test_data_biased200_different_weight.txt'],
    #          ['data/test_data_random200.json', 'data/test_data_random200_different_weight.txt']]
    files = [['data/test_data_biased.json', 'data/test_data_biased_different_weight.txt'],
             ['data/test_data_random.json', 'data/test_data_random_different_weight.txt']]
    test(our, files[0][0], files[0][1])
    print('biased ok')
    test(our, files[1][0], files[1][1])