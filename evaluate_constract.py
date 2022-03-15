import json
import torch
from main_our_v2 import Model as our_model
from main_lost import Model as lost_model
from main_lost_persona import Model as lost_persona_model
from main_transfer_persona import Model as transfer_persona_model
from main_transfer import Model as transfer_model
from main_origin import Model as seq2seq_model

def test(seq2seq, lost, transfer, our, lost_persona, transfer_persona, input_file, output_file):
    with open(output_file, 'w', encoding='utf8') as fw:
        with open(input_file, 'r', encoding='utf8') as fr:
            lines = fr.readlines()
            weights = [torch.Tensor([[1, 0]]), torch.Tensor([[0, 1]])]
            weights = [i.to('cuda') for i in weights]
            for line in lines[:10]:
                line = line.strip('\n')
                data = json.loads(line)
                dialog = data['dialog']
                uid = data['uid']
                profile_all = data['profile']
                if 'responder_profile' in data:
                    responder_profile = data['responder_profile']
                else:
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
                # seq2seq predict
                ans_auto = seq2seq.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                fw.write('predict with seq2seq: ' + ans_auto + '\n')
                # lost predict
                ans_auto = lost.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                fw.write('predict with lost in conversation: ' + ans_auto + '\n')
                # lost persona predict
                ans_auto = lost_persona.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                fw.write('predict with persona lost in conversation: ' + ans_auto + '\n')
                # transfer predict
                ans_auto = transfer.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                fw.write('predict with transfertransfo: ' + ans_auto + '\n')
                # transfer persona predict
                ans_auto = transfer_persona.gen_response([data])
                ans_auto = ''.join(ans_auto[0])
                fw.write('predict with persona transfertransfo: ' + ans_auto + '\n')
                # our model predict
                ans_auto = our.gen_response([data], human=True)
                ans_auto = ''.join(ans_auto[0])
                ans_persona = our.gen_response([data], weight_i=weights[0], human=True)
                ans_persona = ''.join(ans_persona[0])
                ans_nopersona = our.gen_response([data], weight_i=weights[1], human=True)
                ans_nopersona = ''.join(ans_nopersona[0])
                fw.write('predict with auto weight: ' + ans_auto + '\n')
                fw.write('predict with full persona: ' + ans_persona + '\n')
                fw.write('predict with no persona: ' + ans_nopersona + '\n')
                fw.write('\n')

if __name__ == '__main__':
    our = our_model()
    lost = lost_model()
    transfer = transfer_model()
    seq2seq = seq2seq_model()
    lost_persona = lost_persona_model()
    transfer_persona = transfer_persona_model()
    files = [['data/test_data_biased.json', 'data/test_data_biased_contrast.txt'],
             ['data/test_data_random.json', 'data/test_data_random_contrast.txt']]
    test(seq2seq, lost, transfer, our, lost_persona, transfer_persona, files[0][0], files[0][1])
    # test(seq2seq, lost, transfer, our, lost_persona, transfer_persona, files[1][0], files[1][1])