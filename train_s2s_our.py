import random
import torch
from model.utils import load_openai_weights_chinese, set_seed, f1_score
from model.transformer_model_s2s_soft import TransformerSoftModel
from model.trainer_s2s_soft import TrainerSoft
from model.text import myVocab
from model.dataset import SMPDataset_our
from config import get_model_config_our, get_trainer_config_our
import torch.nn as nn
import os
import re
from torch.nn.parallel import DistributedDataParallel
import argparse
import warnings

# torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")
cop = re.compile("[^0-9]")
torch.backends.cudnn.deterministic = True


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def main():
    model_config = get_model_config_our()
    trainer_config = get_trainer_config_our()

    set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)
    # zrs
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    # distributed = (args.local_rank != -1)
    distributed = False
    if distributed:
        print(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    vocab = myVocab(model_config.vocab_path)

    transformer = TransformerSoftModel(n_layers=model_config.n_layers,
                                       n_embeddings=len(vocab),
                                       n_pos_embeddings=model_config.n_pos_embeddings,
                                       embeddings_size=model_config.embeddings_size,
                                       padding_idx=vocab.pad_id,
                                       n_heads=model_config.n_heads,
                                       dropout=model_config.dropout,
                                       embed_dropout=model_config.embed_dropout,
                                       attn_dropout=model_config.attn_dropout,
                                       ff_dropout=model_config.ff_dropout,
                                       bos_id=vocab.bos_id,
                                       eos_id=vocab.eos_id,
                                       max_seq_len=model_config.max_seq_len,
                                       beam_size=model_config.beam_size,
                                       length_penalty=model_config.length_penalty,
                                       n_segments=model_config.n_segments,
                                       annealing_topk=model_config.annealing_topk,
                                       temperature=model_config.temperature,
                                       annealing=model_config.annealing,
                                       diversity_coef=model_config.diversity_coef,
                                       diversity_groups=model_config.diversity_groups,
                                       n_gender=3,
                                       n_loc=256,  ## 越界了?? 问题出在LOC???
                                       n_tag=502)

    if not trainer_config.load_last:
        openai_model = torch.load(trainer_config.openai_parameters_dir, map_location=device)
        openai_model_new = {v: openai_model[v] for k, v in enumerate(openai_model) if not v.endswith('.attn.bias')}
        del openai_model_new['lm_head.weight']
        print(len(openai_model_new.keys()))
        model_dict = {}
        tag = 0
        for idx1, key1 in enumerate(transformer.transformer_module.state_dict()):
            try:
                for idx2, key2 in enumerate(openai_model_new):
                    if idx1 == idx2 and idx1 != 2 and idx1 != 3 and idx1 != 4:
                        if transformer.transformer_module.state_dict()[key1].shape != openai_model[key2].shape:
                            model_dict[key1] = openai_model[key2].transpose(1, 0)
                        else:
                            model_dict[key1] = openai_model[key2]
                    if idx1 == 2:
                        model_dict['gender_embeddings.weight'] = transformer.transformer_module.state_dict()[
                            'gender_embeddings.weight']
                        break
                    if idx1 == 3:
                        model_dict['loc_embeddings.weight'] = transformer.transformer_module.state_dict()[
                            'loc_embeddings.weight']
                        break
                    if idx1 == 4:
                        model_dict['tag_embeddings.weight'] = transformer.transformer_module.state_dict()[
                            'tag_embeddings.weight']
                        tag = 1
                        break
                    if idx1 == (idx2 + 3) and tag == 1:
                        if transformer.transformer_module.state_dict()[key1].shape != openai_model[key2].shape:
                            model_dict[key1] = openai_model[key2].transpose(1, 0)
                            break
                        else:
                            model_dict[key1] = openai_model[key2]
                            break
            except:
                pass
        assert (len(transformer.transformer_module.state_dict().keys()) == len(model_dict))
        transformer.transformer_module.load_state_dict(model_dict)
        print('OpenAI weights chinese loaded from {}'.format(trainer_config.openai_parameters_dir))

    train_dataset = SMPDataset_our(trainer_config.train_datasets, vocab, transformer.n_pos_embeddings - 1, is_test=False)
    test_dataset = SMPDataset_our(trainer_config.test_datasets, vocab, transformer.n_pos_embeddings -1 , is_test=True)

    model_trainer = TrainerSoft(transformer,
                                train_dataset,
                                test_dataset,
                                batch_size=trainer_config.batch_size,
                                batch_split=trainer_config.batch_split,
                                lr=trainer_config.lr,
                                lr_warmup=trainer_config.lr_warmup,
                                lm_weight=trainer_config.lm_weight,
                                risk_weight=trainer_config.risk_weight,
                                n_jobs=trainer_config.n_jobs,
                                clip_grad=trainer_config.clip_grad,
                                device=device,
                                ignore_idxs=vocab.special_tokens_ids,
                                distributed=distributed)
    if distributed:
        model_trainer.model.transformer_module = DistributedDataParallel(model_trainer.model.transformer_module,
                                                                         device_ids=[args.local_rank],
                                                                         output_device=args.local_rank)
        model_trainer.model.cls_linear = DistributedDataParallel(model_trainer.model.cls_linear,
                                                                 device_ids=[args.local_rank],
                                                                 output_device=args.local_rank)

    start_epoch = 0
    init_epoch = 0  ##plus one
    if trainer_config.load_last:
        state_dict = torch.load(trainer_config.last_checkpoint_path + str(init_epoch - 1), map_location=device)
        model_trainer.load_state_dict(state_dict)
        # start_epoch = int(cop.sub('', trainer_config.last_checkpoint_path.split('/')[-1])) + 1
        start_epoch = init_epoch
        print('Weights loaded from {}'.format(trainer_config.last_checkpoint_path + str(init_epoch - 1)))

    # helpers -----------------------------------------------------
    def save_func(epoch):
        dirs = '/'.join(trainer_config.last_checkpoint_path.split('/')[:-1])  # decide which checkpoints to save
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        torch.save(model_trainer.state_dict(), trainer_config.last_checkpoint_path)
        torch.save(model_trainer.state_dict(), trainer_config.last_checkpoint_path + str(epoch))
        if os.path.exists(trainer_config.last_checkpoint_path + str(epoch - 30)):
            os.remove(trainer_config.last_checkpoint_path + str(epoch - 30))

    def sample_text_func(epoch):
        n_samples = 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for persona, dialog, target, gender, loc, tag, label in samples:
            contexts = [torch.tensor([c], dtype=torch.long, device=model_trainer.device) for c in [persona, dialog] if
                        len(c) > 0]
            gender_format = torch.tensor([gender], dtype=torch.long, device=model_trainer.device)
            loc_format = torch.tensor([loc], dtype=torch.long, device=model_trainer.device)
            tag_format = torch.tensor([tag], dtype=torch.long, device=model_trainer.device)
            persona_enc = model_trainer.model.encode(contexts[0])
            dialog_enc = model_trainer.model.encode(contexts[1], gender=gender_format, loc=loc_format, tag=tag_format)
            enc_contexts = [persona_enc, dialog_enc]
            weight = model_trainer.model.compute_weight(enc_contexts[1])
            prediction = model_trainer.model.beam_search(enc_contexts, weight=weight)[0]
            context_str = vocab.ids2string(persona[1:-1])
            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.spl, '\n\t- ')
            target_str = vocab.ids2string(target[1:-1])
            prediction_str = vocab.ids2string(prediction)
            print('\n')
            print('weight: ', weight)
            print('label: ', label)
            print('Persona info:\n\t{}'.format(context_str))
            print('Dialog:{}'.format(dialog_str))
            print('Target:\n\t{}'.format(target_str))
            print('Prediction:\n\t{}'.format(prediction_str))

    def test_func(epoch):
        if (epoch + 1) % trainer_config.test_period == 0:
            metric_funcs = {'f1_score': f1_score}
            model_trainer.test(metric_funcs)

    def f1_risk(predictions, targets):
        scores = f1_score(predictions, targets, average=False)
        return [1 - s for s in scores]

        # helpers -----------------------------------------------------

    # model_trainer.model.transformer_module = nn.DataParallel(model_trainer.model.transformer_module, device_ids=[0, 1])
    try:
        if args.local_rank in [-1, 0]:
            model_trainer.train(start_epoch, trainer_config.n_epochs,
                                after_epoch_funcs=[save_func, sample_text_func, test_func],
                                # after_epoch_funcs=[sample_text_func, test_func],
                                risk_func=f1_risk)
        else:
            model_trainer.train(start_epoch, trainer_config.n_epochs)
        # model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[sample_text_func], risk_func=f1_risk)
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(model_trainer.state_dict(), trainer_config.interrupt_checkpoint_path)
        raise e


if __name__ == '__main__':
    main()
