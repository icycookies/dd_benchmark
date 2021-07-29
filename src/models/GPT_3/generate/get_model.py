import os
import torch
import random
from arguments import get_args
from utils import load_checkpoint, get_checkpoint_iteration
from pretrain_gpt2 import initialize_distributed, get_masks_and_position_ids, set_random_seed
import mpu
from data_utils import make_tokenizer
from utils import print_rank_0
from pypinyin import pinyin, TONE3
from pretrain_gpt2 import get_model


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)

    # if args.deepspeed:
    #     print_rank_0("DeepSpeed is enabled.")
    #
    #     model, _, _, _ = deepspeed.initialize(
    #         model=model,
    #         model_parameters=model.parameters(),
    #         args=args,
    #         mpu=mpu,
    #         dist_init_required=False
    #     )
    if args.load is not None:
        if args.deepspeed:
            iteration, release, success = get_checkpoint_iteration(args)
            print(iteration)
            path = os.path.join("/home/huangs/zhipu/model/poem_0702/166000", "mp_rank_00_model_states.pt")
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["module"])
        else:
            _ = load_checkpoint(
                model, None, None, args, load_optimizer_states=False)
    # if args.deepspeed:
    #     model = model.module

    return model

def set_args():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
    # set up
    # print(args)
    args.deepspeed = True
    args.num_nodes = 1
    args.num_gpus = 1
    args.model_parallel_size = 1
    args.deepspeed_config = "script_dir/ds_config.json"
    args.num_layers = 32
    args.hidden_size = 2560
    args.load = "/mnt3/ckp"
    # args.load="/mnt3/ckp/checkpoint2/new"
    args.num_attention_heads = 32
    args.max_position_embeddings = 1024
    args.tokenizer_type = "ChineseSPTokenizer"
    args.cache_dir = "cache"
    args.fp16 = True
    args.out_seq_length = 180
    args.seq_length = 200
    args.mem_length = 256
    args.transformer_xl = True
    args.temperature = 1
    args.top_k = 0
    args.top_p = 0

    return args

def prepare_tokenizer(args):
    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    tokenizer = make_tokenizer(**tokenizer_args)

    num_tokens = tokenizer.num_tokens
    before = num_tokens
    after = before
    multiple = args.make_vocab_size_divisible_by * \
               mpu.get_model_parallel_world_size()
    while (after % multiple) != 0:
        after += 1
    print_rank_0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(
        before, after - before, after))

    args.tokenizer_num_tokens = after
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    # after = tokenizer.num_tokens
    # while after % mpu.get_model_parallel_world_size() != 0:
    #     after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer

def get_batch(context_tokens, device, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    print(tokens)
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        transformer_xl=args.transformer_xl,
        mem_length=args.mem_length)
    return tokens, attention_mask, position_ids

def prepare_model():
    """Main training program."""

    # print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = set_args()
    # print(args)
    args.mem_length = args.seq_length + args.mem_length - 1

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)
    # args.load="../ckp/txl-2.8b11-20-15-10"
    # model2=setup_model(args)
    # setting default batch size to 1
    args.batch_size = 1

    # generate samples
    return model, tokenizer, args

def get_feature(model, tokenizer, args, device, content, max_tries):
    input_str = content
    input_str1 = "草泥马"
    model.eval()
    with torch.no_grad():
        # index=len(tokens[0])
        mct_tree = []
        context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
        eo_tokens = tokenizer.EncodeAsIds('<|endoftext|>').tokenization
        context_length = len(context_tokens)
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        eo_token_tensor = torch.cuda.LongTensor(eo_tokens)
        context_length_tensor = torch.cuda.LongTensor([context_length])
        context_length = context_length_tensor[0].item()
        # tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)

        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)
        print(tokens.shape)
        print(attention_mask.shape)
        print(position_ids)
        print(type(model))
        logits, *rts = model(tokens, position_ids, attention_mask)
        print(logits.shape)
        print(rts[0].shape)
        

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
content = "伞兵一号卢本伟准备就绪"
model, tokenizer, args = prepare_model()
get_feature(model, tokenizer, args, torch.cuda.current_device(), content, 100)