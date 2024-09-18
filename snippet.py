import argparse
import six
from vocab import Vocab
from dataloader import ABSA_Dataset, ABSA_DataLoader, ABSA_collate_fn

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default='./data/V2/MAMS')
    parser.add_argument('--vocab_dir', type=str, default='./data/V2/MAMS')
    parser.add_argument('--data_name', type=str, default='mams')
    parser.add_argument('--best_model_dir', type=str, default='./best_mod]el_checkpoints/best_bert/')

    parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')

    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='bert dim.')
    parser.add_argument('--lstm_dim', type=int, default=300, help="dimension of bi-lstm")
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument("--gamma", default=0.1, type=float, help="The balance of adaptive loss.")
    parser.add_argument("--alpha", default=0.15, type=float, help="The balance of span loss.")
    parser.add_argument("--beta", default=0.7, type=float, help="The balance of root loss.")
    parser.add_argument("--dep_type_size",default=42,type=float, help="The length of dependency type dict.")
    parser.add_argument("--dep_embed_dim", default=25, type=int, help="The dimension of dependency type .")


    parser.add_argument('--input_dropout', type=float, default=0.2, help='input dropout rate.')
    parser.add_argument('--layer_dropout', type=float, default=0.1, help='layer dropout rate.')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='self-attention layer dropout rate.')
    parser.add_argument('--bert_dropout',type=float,default=0.1,help='bert dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='gcn dropout rate.')
    parser.add_argument('--gcn_layers', type=int, default=3, help='gcn num layers.')
    parser.add_argument('--num_layers', type=int, default=2, help='module num layers.')


    parser.add_argument('--lower', default=True, help = 'lowercase all words.')
    parser.add_argument('--need_preprocess', default=False, help = 'need parse data.')

    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate.')
    parser.add_argument('--bert_lr', type=float, default=2e-5, help='learning rate for bert.')
    parser.add_argument('--l2', type=float, default=1e-5, help='weight decay rate.')

    parser.add_argument('--num_encoder_layer', type=int, default=3, help='Number of graph layers.')
    parser.add_argument('--num_epoch', type=int, default=20, help='Number of total training epochs.')
    parser.add_argument('--max_patience', type=int, default=20, help='max patience in training')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--log_step', type=int, default=16, help='Print log every k steps.')

    parser.add_argument('--seed', type=int, default= 1)
    parser.add_argument('--max_len', type=int, default=100)

    parser.add_argument('--average_mapback', default=False, action='store_true')

    parser.add_argument('--leaf2root', default=False, action='store_true')
    parser.add_argument('--root2leaf', default=False, action='store_true')

    parser.add_argument('--con_dep_version', default='wo_dep', type=str)
    parser.add_argument('--losstype', type=str, default='None')
    parser.add_argument('--con_dep_conditional', default=False, action = 'store_true')

    parser.add_argument('--dynamic_tree_attn_head', type=int, default=4)
    parser.add_argument('--fusion_attention_heads', type=int, default=6)
    parser.add_argument('--attention_heads', type=int, default=4)
    parser.add_argument('--max_num_spans', type=int, default=4)
    parser.add_argument('--special_token', default='[N]')
    parser.add_argument('--adj_span_version', type=int, default=0)

    parser.add_argument('--sort_key_idx', default=0, help='sort idx')
    


    args = parser.parse_args()

    return args






def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

def totally_parameters(model):  #
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

def load_vocab(args):
    print('Loading vocab...')

    vocab = {
        'token': Vocab.load_vocab(args.vocab_dir + '/vocab_tok.vocab'),
        'polarity': Vocab.load_vocab(args.vocab_dir + '/vocab_pol.vocab')
    }

    print(
        'token_vocab: {}, polarity_vocab:{}'.format(len(vocab['token']), len(vocab['polarity']))
    )

    args.tok_size = len(vocab['token'])
    return vocab


def load_one_data(args, file_name, vocab, tokenizer, block_shuffle = True, is_shuffle=True,flag="train"):
    print('Loading data from {} with batch size {}...'.format(file_name, args.batch_size))
    one_dataset = ABSA_Dataset(args, file_name, vocab, tokenizer,flag)

    if block_shuffle and is_shuffle:
        one_dataloader = ABSA_DataLoader(one_dataset, 
                                        sort_key = lambda x: x[args.sort_key_idx],
                                        is_shuffle = is_shuffle,
                                        batch_size = args.batch_size,
                                        collate_fn = ABSA_collate_fn
                                        )
    else:
        one_sampler = RandomSampler(one_dataset) if is_shuffle else SequentialSampler(one_dataset)

        one_dataloader = DataLoader(one_dataset, 
                                    sampler=one_sampler,
                                    batch_size=args.batch_size,
                                    collate_fn = ABSA_collate_fn)
    return one_dataloader

def load_data(args, vocab, tokenizer=None):
    train_dataloader = load_one_data(args, file_name = args.data_dir + '/train_con_new.json',
                                     vocab = vocab, tokenizer = tokenizer, is_shuffle = True,flag="train")

    valid_dataloader = load_one_data(args, file_name = args.data_dir + '/valid_con_new.json',
                                     vocab = vocab, tokenizer = tokenizer, is_shuffle = False,flag="valid")

    test_dataloader = load_one_data(args, file_name = args.data_dir + '/test_con_new.json',
                                     vocab = vocab, tokenizer = tokenizer, is_shuffle = False,flag="test")
    
    return train_dataloader, valid_dataloader, test_dataloader

