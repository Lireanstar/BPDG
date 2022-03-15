import json
import torch
# from main_our import Model as our_model
from main_our_v2 import Model as our_model
from main_lost import Model as lost_model
from main_lost_persona import Model as lost_persona_model
from main_transfer_persona import Model as transfer_persona_model
from main_transfer import Model as transfer_model
from main_origin import Model as seq2seq_model

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

def test(seq2seq, lost, transfer, our, lost_persona, transfer_persona, input_file, output_file):
    import tqdm
    with open(output_file, 'w', encoding='utf8') as fw:
        with open(input_file, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            weights = [torch.Tensor([[1, 0]]), torch.Tensor([[0, 1]])]
            weights = [i.to('cuda') for i in weights]
            cnt = 0
            for line in tqdm.tqdm(lines[:10]):
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
                # seq2seq predict
                ans_auto = seq2seq.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                fw.write('transformer' + '\t' + ans_auto + '\n')
                # lost predict
                ans_auto = lost.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                fw.write('lost' + '\t' + ans_auto + '\n')
                # lost persona predict
                ans_auto = lost_persona.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                fw.write('persona lost' + '\t' + ans_auto + '\n')
                # transfer predict
                ans_auto = transfer.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                fw.write('transfer' + '\t' + ans_auto + '\n')
                # transfer persona predict
                ans_auto = transfer_persona.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                fw.write('persona transfer' + '\t' + ans_auto + '\n')
                # our model predict
                ans_auto = our.gen_response([data], human=True)
                ans_auto = ''.join(ans_auto[0])
                ans_persona = our.gen_response([data], weight_i=weights[0], human=True)
                ans_persona = ''.join(ans_persona[0])
                ans_nopersona = our.gen_response([data], weight_i=weights[1], human=True)
                ans_nopersona = ''.join(ans_nopersona[0])
                fw.write('our with auto' + '\t' + ans_auto + '\n')
                fw.write('our with full' + '\t' + ans_persona + '\n')
                fw.write('our with none' + '\t' + ans_nopersona + '\n')
                fw.write('\n\n\n\n')

if __name__ == '__main__':
    our = our_model()
    lost = lost_model()
    transfer = transfer_model()
    seq2seq = seq2seq_model()
    lost_persona = lost_persona_model()
    transfer_persona = transfer_persona_model()
    files = [['data/test_data_biased.json', 'data/test_data_biased200_human.txt'],
             ['data/test_data_random.json', 'data/test_data_random200_human.txt']]
    test(seq2seq, lost, transfer, our, lost_persona, transfer_persona, files[0][0], files[0][1])
    print('biased ok')
    # test(seq2seq, lost, transfer, our, lost_persona, transfer_persona, files[1][0], files[1][1])