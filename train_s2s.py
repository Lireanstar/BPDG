import torch
import random
from model.utils import load_openai_weights_chinese, set_seed, f1_score
from model.transformer_context_model import TransformerContextModel
from model.trainer_context import TrainerContext
from model.text import myVocab
from model.dataset import SMPDataset
from config import get_model_config_context, get_trainer_config_context
import torch.nn as nn
import os
import re

cop = re.compile("[^0-9]")
import warnings

warnings.filterwarnings("ignore")


def main():
    model_config = get_model_config_context()
    trainer_config = get_trainer_config_context()

    # set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)

    vocab = myVocab(model_config.vocab_path)

    transformer = TransformerContextModel(n_layers=model_config.n_layers,
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
                                          diversity_groups=model_config.diversity_groups)

    # if not trainer_config.load_last:
    #     openai_model = torch.load(trainer_config.openai_parameters_dir, map_location=device)
    #     openai_model.pop('decoder.pre_softmax.weight')
    #     b = list(openai_model.keys())
    #     for i in b:
    #         openai_model[i.split('.', 1)[1]] = openai_model.pop(i)
    #     transformer.transformer_module.load_state_dict(openai_model)
    #     # load_openai_weights_chinese(transformer.transformer_module, trainer_config.openai_parameters_dir)
    #     print('OpenAI weights chinese loaded from {}'.format(trainer_config.openai_parameters_dir))
    load_last = False
    if not load_last:
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


    train_dataset = SMPDataset(trainer_config.train_datasets, vocab, transformer.n_pos_embeddings - 1)
    test_dataset = SMPDataset(trainer_config.test_datasets, vocab, transformer.n_pos_embeddings - 1)

    model_trainer = TrainerContext(transformer,
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
                                   ignore_idxs=vocab.special_tokens_ids)

    start_epoch = 0
    init_epoch = 123
    # if trainer_config.load_last:
    #     state_dict = torch.load(trainer_config.last_checkpoint_path + str(init_epoch - 1), map_location=device)
    #     model_trainer.load_state_dict(state_dict)
    #     # start_epoch = int(cop.sub('', trainer_config.last_checkpoint_path.split('/')[-1])) + 1
    #     start_epoch = init_epoch
    #     print('Weights loaded from {}'.format(trainer_config.last_checkpoint_path + str(init_epoch - 1)))

    # helpers -----------------------------------------------------
    def save_func(epoch):
        torch.save(model_trainer.state_dict(), trainer_config.last_checkpoint_path)
        torch.save(model_trainer.state_dict(), trainer_config.last_checkpoint_path + str(epoch))
        if os.path.exists(trainer_config.last_checkpoint_path + str(epoch - 30)):
            os.remove(trainer_config.last_checkpoint_path + str(epoch - 30))

    def sample_text_func(epoch):
        n_samples = 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for context, dialog, target in samples:
            contexts = [torch.tensor([c], dtype=torch.long, device=model_trainer.device) for c in [context, dialog] if
                        len(c) > 0]
            prediction = model_trainer.model.predict(contexts)[0]
            context_str = vocab.ids2string(context[1:-1])
            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.spl, '\n\t- ')
            target_str = vocab.ids2string(target[1:-1])
            prediction_str = vocab.ids2string(prediction)

            print('\n')
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
        model_trainer.train(start_epoch, trainer_config.n_epochs,
                            after_epoch_funcs=[save_func, sample_text_func, test_func],
                            risk_func=f1_risk)
        # model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[sample_text_func], risk_func=f1_risk)
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(model_trainer.state_dict(), trainer_config.interrupt_checkpoint_path)
        raise e


if __name__ == '__main__':
    main()
