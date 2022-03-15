import torch
import random
import os
from model.utils import load_openai_weights_chinese, set_seed, f1_score
from model.transformer_model_lm_transfer import TransformerTransferModel
from model.trainer_lm_transfer import TrainerTransfer
from model.text import myVocab
from model.dataset import SMPDataset_transfer
from config import get_model_config_transfer, get_trainer_config_transfer
import torch.nn as nn
import re
from torch.nn.parallel import DistributedDataParallel
import argparse
cop = re.compile("[^0-9]")


def main():
    model_config = get_model_config_transfer()
    trainer_config = get_trainer_config_transfer()

    set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)
    # zrs
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    distributed = (args.local_rank != -1)
    if distributed:
        print(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    vocab = myVocab(model_config.vocab_path)

    transformer = TransformerTransferModel(n_layers=model_config.n_layers,
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
                                     annealing=model_config.annealing,
                                     diversity_coef=model_config.diversity_coef,
                                     diversity_groups=model_config.diversity_groups)

    if not trainer_config.load_last:
        openai_model = torch.load(trainer_config.openai_parameters_dir, map_location=device)
        # openai_model.pop('decoder.pre_softmax.weight')
        b = list(openai_model.keys())
        # print(b)
        model_dict = {}
        for idx1, key1 in enumerate(transformer.transformer_module.state_dict()):
            try:
                for idx2, key2 in enumerate(openai_model):
                    if idx1 == idx2:
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

    train_dataset = SMPDataset_transfer(trainer_config.train_datasets, vocab, transformer.n_pos_embeddings - 1)
    test_dataset = SMPDataset_transfer(trainer_config.test_datasets, vocab, transformer.n_pos_embeddings - 1)

    model_trainer = TrainerTransfer(transformer,
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
                                                                         device_ids=[args.local_rank], output_device=args.local_rank)

    start_epoch = 0
    init_epoch = 0
    if trainer_config.load_last:
        state_dict = torch.load(trainer_config.last_checkpoint_path + str(init_epoch - 1), map_location=device)
        model_trainer.load_state_dict(state_dict)
        # start_epoch = int(cop.sub('', trainer_config.last_checkpoint_path.split('/')[-1])) + 1
        start_epoch = init_epoch
        print('Weights loaded from {}'.format(trainer_config.last_checkpoint_path + str(init_epoch - 1)))

    # helpers -----------------------------------------------------
    def save_func(epoch):
        dirs = '/'.join(trainer_config.last_checkpoint_path.split('/')[:-1])
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        torch.save(model_trainer.state_dict(), trainer_config.last_checkpoint_path)
        torch.save(model_trainer.state_dict(), trainer_config.last_checkpoint_path + str(epoch))
        if os.path.exists(trainer_config.last_checkpoint_path + str(epoch-10)):
            os.remove(trainer_config.last_checkpoint_path + str(epoch-10))

    def sample_text_func(epoch):
        n_samples = 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for source, target, id_user in samples:
            # spl_idx = [i for i, v in enumerate(target) if v == vocab.p1_id or v == vocab.p2_id]
            spl_idx = [i for i, v in enumerate(source) if v == vocab.spl_id]
            last_spl = spl_idx[-1]
            prefix = source[1:last_spl+1]
            prefix_id = id_user[1:last_spl+1]
            prediction = model_trainer.model.beam_search(prefix=prefix, prefix_id=prefix_id)[0]
            target_str = vocab.ids2string(source[1:-1])
            prediction_str = vocab.ids2string(prediction)
            print('\n')
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
            model_trainer.train(start_epoch, trainer_config.n_epochs, after_epoch_funcs=[save_func, sample_text_func, test_func],
                                risk_func=f1_risk)
        else:
            model_trainer.train(start_epoch, trainer_config.n_epochs)
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(model_trainer.state_dict(), trainer_config.interrupt_checkpoint_path)
        raise e


if __name__ == '__main__':
    main()