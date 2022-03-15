from attrdict import AttrDict
from model.utils import openai_transformer_config


# transformer config
def get_model_config_context():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 3,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 2.2,
                       'n_segments': None})

    return config


def get_test_config_context():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': './parameters/last_checkpoint'})

    return config


def get_test_config_context_ensemble():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': ['./data/baidu_best_checkpoint_479.pt']})
    return config


def get_trainer_config_context():
    config = AttrDict({'n_epochs': 500,
                       'batch_size': 16,
                       'batch_split': 1,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.2,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       # 'load_last': True,
                       'openai_parameters_dir': './parameters/pytorch_model.bin',
                       'last_checkpoint_path': './checkpoints/our-v3/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/our-v3/interrupt_checkpoint',
                       # 'train_datasets': ['./data/our/train_data_label.json'],
                       'train_datasets': ['./data/data_v2/90.json'],
                       # 'train_datasets': ['./data/test_data_random.json'],
                       # 'test_datasets': ['./data/our/test_data_label.json']})
                       'test_datasets': ['./data/data_v2/90.json']})
    return config


# transformer config our
def get_model_config_our():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 3,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 2.2,
                       'n_segments': None})

    return config


def get_trainer_config_our():
    config = AttrDict({'n_epochs': 500,
                       'batch_size': 8,
                       'batch_split': 1,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.2,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       # 'load_last': True,
                       # 'openai_parameters_dir': './parameters/pytorch_model.bin',
                       'openai_parameters_dir': './parameters/LCCD_GPT/pytorch_model.bin',
                       'last_checkpoint_path': './checkpoints/our-v2/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/our-v2/interrupt_checkpoint',
                       # 'train_datasets': ['./data/our/train_data_label.json'],
                       # 'train_datasets': ['./data/data_v2/data_v3/train_data_label.json'],
                       'train_datasets': ['./data/data_v2/data_v3/merged_train_dialogue.json'],
                       # 'train_datasets': ['./data/test_data_random.json'],
                       # 'test_datasets': ['./data/our/test_data_label.json']})
                       # 'test_datasets': ['./data/data_v2/data_v3/test_data_label.json']})
                       'test_datasets': ['./data/data_v2/data_v3/merged_test_dialogue.json']})
    return config


def get_test_config_our():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       # 'last_checkpoint_path': '/home/data/zhengyinhe/AAAI_personachat/checkpoints/our-v2/last_checkpoint30'})
                       'last_checkpoint_path': './checkpoints/our-v2/last_checkpoint'})
    return config


# transformer config soft
def get_model_config_soft():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 3,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 2.2,
                       'n_segments': None})

    return config


def get_trainer_config_soft():
    config = AttrDict({'n_epochs': 500,
                       'batch_size': 8,
                       'batch_split': 1,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.2,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       # 'openai_parameters_dir': './parameters/pytorch_model.bin',
                       'openai_parameters_dir': './parameters/LCCD_GPT/pytorch_model.bin',
                       'last_checkpoint_path': './checkpoints/soft/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/soft/interrupt_checkpoint',
                       # 'train_datasets': ['./data/soft/train_data_label.json'],
                       # 'train_datasets': ['./data/data_v2/train_data_label.json'],
                       # 'train_datasets': ['./data/data_v2/data_v3/train_data_label.json'],
                       'train_datasets': ['./data/data_v2/data_v3/merged_train_dialogue.json'],
                       # 'test_datasets': ['./data/soft/test_data_label.json']})
                       # 'test_datasets': ['./data/data_v2/test_data_label.json']})
                       # 'test_datasets': ['./data/data_v2/data_v3/test_data_label.json']})
                       'test_datasets': ['./data/data_v2/data_v3/merged_test_dialogue.json']})

    return config


def get_test_config_soft():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': './checkpoints/soft/last_checkpoint0'})
    return config


# transformer config lost
def get_model_config_lost():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 3,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 2.2,
                       'n_segments': None})

    return config


def get_trainer_config_lost():
    config = AttrDict({'n_epochs': 500,
                       'batch_size': 256,
                       'batch_split': 32,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.2,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': True,
                       'openai_parameters_dir': './parameters/pytorch_model.bin',
                       'last_checkpoint_path': './checkpoints/lost/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/lost/interrupt_checkpoint',
                       'train_datasets': ['./data/lost/train_data_label.json'],
                       'test_datasets': ['./data/lost/test_data_label.json']})

    return config


def get_test_config_lost():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': './checkpoints/lost/last_checkpoint0'})
    return config


# transformer config transfer
def get_model_config_transfer():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 5,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 2.3,
                       'n_segments': None})

    return config


def get_trainer_config_transfer():
    config = AttrDict({'n_epochs': 500,
                       'batch_size': 256,
                       'batch_split': 32,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.2,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       'openai_parameters_dir': './parameters/pytorch_model.bin',
                       'last_checkpoint_path': './checkpoints/transfer/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/transfer/interrupt_checkpoint',
                       'train_datasets': ['./data/transfer/train_data_label.json'],
                       'test_datasets': ['./data/transfer/test_data_label.json']})

    return config


def get_test_config_transfer():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': './checkpoints/transfer/last_checkpoint0'})
    return config


# transformer config our unembedding
def get_model_config_unembedding():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 5,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 2.2,
                       'n_segments': None})

    return config


def get_trainer_config_unembedding():
    config = AttrDict({'n_epochs': 500,
                       'batch_size': 256,
                       'batch_split': 32,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.2,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       # 'openai_parameters_dir': './parameters/chinese_pretrain.pt',
                       'openai_parameters_dir': './parameters/pytorch_model.bin',
                       'last_checkpoint_path': './checkpoints/unembedding-v2/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/unembedding-v2/interrupt_checkpoint',
                       'train_datasets': ['./data/unembedding/train_data_label.json'],
                       'test_datasets': ['./data/unembedding/test_data_label.json']})

    return config


def get_test_config_unembedding():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': './checkpoints/unembedding-v2/last_checkpoint'})
    return config


# transformer config unweight
def get_model_config_unweight():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 5,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 2.2,
                       'n_segments': None})

    return config


def get_trainer_config_unweight():
    config = AttrDict({'n_epochs': 500,
                       'batch_size': 512,
                       'batch_split': 16,
                       'lr': 6.25e-7,
                       'lr_warmup': 1000,
                       'lm_weight': 0.2,
                       'risk_weight': 0,
                       'n_jobs': 0,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       'openai_parameters_dir': './parameters/pytorch_model.bin',
                       'last_checkpoint_path': './checkpoints/unweight/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/unweight/interrupt_checkpoint',
                       'train_datasets': ['./data/lost/train_data_label.json'],
                       'test_datasets': ['./data/lost/test_data_label.json']})

    return config


def get_test_config_unweight():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': './checkpoints/unweight/last_checkpoint'})
    return config


# transformer config s2s origin
def get_model_config_origin():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 5,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 2.3,
                       'n_segments': None})

    return config


def get_trainer_config_origin():
    config = AttrDict({'n_epochs': 500,
                       'batch_size':16,
                       'batch_split': 1,
                       'lr': 6.25e-7,
                       'lr_warmup': 1000,
                       'lm_weight': 0.2,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       # 'openai_parameters_dir': './parameters/chinese_pretrain.pt',
                       # 'openai_parameters_dir': './parameters/pytorch_model.bin',
                       'openai_parameters_dir': './parameters/LCCD_GPT/pytorch_model.bin',
                       'last_checkpoint_path': './checkpoints/origin/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/origin/interrupt_checkpoint',
                       # 'train_datasets': ['./data/lost/train_data_label.json'],
                       # 'test_datasets': ['./data/lost/test_data_label.json']})
                        'train_datasets': ['./data/data_v2/train_data_label.json'],
                        'test_datasets': ['./data/data_v2/test_data_label.json']})

    return config


def get_test_config_origin():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': './checkpoints/origin/last_checkpoint'})
    return config


# transformer config heuristic
def get_model_config_heuristic():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 3,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 3.5,
                       'n_segments': None})

    return config


def get_trainer_config_heuristic():
    config = AttrDict({'n_epochs': 500,
                       'batch_size': 256,
                       'batch_split': 32,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.2,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       # 'openai_parameters_dir': './parameters/chinese_pretrain.pt',
                       'openai_parameters_dir': './parameters/pytorch_model.bin',
                       'last_checkpoint_path': './checkpoints/heuristic/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/heuristic/interrupt_checkpoint',
                       'train_datasets': ['./data/huristic/train_data_label.json'],
                       # 'test_datasets': ['./train_data/our/valid.json']})
                       'test_datasets': ['./data/huristic/test_data_label.json']})

    return config


def get_test_config_heuristic():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': './checkpoints/heuristic/last_checkpoint'})
    return config


# transformer config unpretrain
def get_model_config_unpretrain():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 5,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 2.2,
                       'n_segments': None})

    return config


def get_trainer_config_unpretrain():
    config = AttrDict({'n_epochs': 500,
                       'batch_size': 20,
                       'batch_split': 1,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.2,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       # 'openai_parameters_dir': './parameters/chinese_pretrain.pt',
                       'openai_parameters_dir': './parameters/pytorch_model.bin',
                       'last_checkpoint_path': './checkpoints/unpretrain-v2/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/unpretrain-v2/interrupt_checkpoint',
                       'train_datasets': ['./data/unpretrain/train_data_label.json'],
                       'test_datasets': ['./data/unpretrain/test_data_label.json']})
    return config


def get_test_config_unpretrain():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': './checkpoints/unpretrain-v2/last_checkpoint'})
    return config


# transformer config lost persona
def get_model_config_lost_persona():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 5,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 2.2,
                       'n_segments': None})

    return config


def get_trainer_config_lost_persona():
    config = AttrDict({'n_epochs': 500,
                       'batch_size': 256,
                       'batch_split': 32,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.2,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       'openai_parameters_dir': './parameters/pytorch_model.bin',
                       # 'openai_parameters_dir': './parameters/chinese_pretrain.pt',
                       'last_checkpoint_path': './checkpoints/lost_persona/last_checkpoint',
                       'interrupt_checkpoint_path': './checkpoints/lost_persona/interrupt_checkpoint',
                       'train_datasets': ['./data/lost_persona/train_data_label.json'],
                       'test_datasets': ['./data/lost_persona/test_data_label.json']})

    return config


def get_test_config_lost_persona():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': './checkpoints/lost_persona/last_checkpoint'})
    return config


# transformer config transfer persona
def get_model_config_transfer_persona():
    default_config = openai_transformer_config()
    config = AttrDict({'vocab_path': './parameters/vocab.txt',
                       'n_layers': default_config.n_layers,
                       'n_pos_embeddings': 512,
                       'embeddings_size': default_config.embeddings_size,
                       'n_heads': default_config.n_heads,
                       'dropout': default_config.dropout,
                       'embed_dropout': default_config.embed_dropout,
                       'attn_dropout': default_config.attn_dropout,
                       'ff_dropout': default_config.ff_dropout,
                       'max_seq_len': 32,
                       'beam_size': 5,
                       'diversity_coef': 0,
                       'diversity_groups': 1,
                       'annealing_topk': 20,
                       'temperature': 0.8,
                       'annealing': 0,
                       'length_penalty': 2.2,
                       'n_segments': None})

    return config


def get_trainer_config_transfer_persona():
    config = AttrDict({'n_epochs': 500,
                       'batch_size': 256,
                       'batch_split': 32,
                       'lr': 6.25e-5,
                       'lr_warmup': 1000,
                       'lm_weight': 0.2,
                       'risk_weight': 0,
                       'n_jobs': 4,
                       'label_smoothing': 0.1,
                       'clip_grad': 1.0,
                       'test_period': 1,
                       'seed': 0,
                       'device': 'cuda',
                       'load_last': False,
                       'openai_parameters_dir': './parameters/pytorch_model.bin',
                       # 'openai_parameters_dir': './parameters/chinese_pretrain.pt',
                       'last_checkpoint_path': './checkpoints/transfer_persona/last_checkpoint',
                       # 'last_checkpoint_path': r'E:\peasona_chatbot\smp_zhangrongsheng\checkpoints\last_checkpoint490',
                       'interrupt_checkpoint_path': './checkpoints/transfer_persona/interrupt_checkpoint',
                       'train_datasets': ['./data/transfer_persona/train_data_label.json'],
                       'test_datasets': ['./data/transfer_persona/test_data_label.json']})

    return config


def get_test_config_transfer_persona():
    config = AttrDict({'seed': 0,
                       'device': 'cuda',
                       'last_checkpoint_path': './checkpoints/transfer_persona/last_checkpoint'})
    return config
