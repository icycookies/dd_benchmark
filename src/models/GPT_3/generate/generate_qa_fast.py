# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from datetime import datetime
from arguments import get_args
from utils import Timers
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from pretrain_gpt2 import get_train_val_test_data
from pretrain_gpt2 import get_masks_and_position_ids
from utils import load_checkpoint, get_checkpoint_iteration
from data_utils import make_tokenizer
from configure_data import configure_data
import mpu
import deepspeed

from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0
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
            iteration = args.iteration
            path = os.path.join(args.load, str(iteration), "mp_rank_00_model_states.pt")
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["module"])
        else:
            _ = load_checkpoint(
                model, None, None, args, load_optimizer_states=False)
    # if args.deepspeed:
    #     model = model.module

    return model


def get_batch(context_tokens, device, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
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


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        #convert to 1D
        logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        #going back to 2D
        logits=logits.view(1, -1).contiguous()
	
    return logits


def getlength(s):
    w=s.replace('。',',').replace('？',',').replace('?',',').replace('!',',').replace('！',',').replace('.',',').replace('；',',').replace(';',',').replace('<\n>',',').replace('<n>',',')
    return len(w)
def end_str(s):
    if "回答" in s:
        s_list = s.split("回答",2)
        if len(s_list) == 2:
            s = s_list[1][1:]
        if len(s_list) == 3:
            s = s_list[2][1:]
    w =s.replace("<\n>","。").replace("<n>","。")
    w += "<|endoftext|>"
    return w



def generate_string(model, tokenizer, args, device,input_str,max_sentence_count=6,max_chars_count=80):
    end_token = [".","。","！","!","?","？","；",";"]
    context_count=0
    model.eval()
    context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
    context_length = len(context_tokens)
    if context_length>=args.seq_length:
        return "输入过长。"
  
    #terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            # pad_id = tokenizer.get_command('pad').Id
            # if context_length < args.out_seq_length:
            #     context_tokens.extend([pad_id] * (args.out_seq_length - context_length))

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor([context_length])
    context_length = context_length_tensor[0].item()

    start_time = time.time()

    counter, mems = 0, []
    org_context_length = context_length
    with torch.no_grad():
        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)
        old_token_list_length = 0
        # decode_counts = 0
        while counter < (args.out_seq_length - org_context_length):
            # decode_counts +=1
            if counter == 0:
                logits, *mems = model(tokens, position_ids, attention_mask, *mems)
            else:
                index = org_context_length + counter
                logits, *mems = model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                              dtype=torch.float), *mems)
            # if decode_counts % 10 == 0:
            #     print("consuming_every_times: " + str(time.time()-b))
            logits = logits[:, -1]
            logits /= args.temperature
            logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
            log_probs = F.softmax(logits, dim=-1)
            prev = torch.multinomial(log_probs, num_samples=1)[0]
            tokens = torch.cat((tokens, prev.view(1, 1)), dim=1)
            context_length += 1
            counter += 1

            output_tokens_list = tokens.view(-1).contiguous()
            decode_tokens = tokenizer.DecodeIds(output_tokens_list.tolist())

            if decode_tokens[-1] in end_token and len(decode_tokens) > max_chars_count:
                sentence_len = getlength(decode_tokens)
                if sentence_len >= max_sentence_count:
                    decode_tokens = end_str(decode_tokens)
                    break

            # 打印每次输出结果
            # print(decode_tokens[old_token_list_length:len(decode_tokens)-1])
            # old_token_list_length = len(decode_tokens) - old_token_list_length
            # if "<|endoftext|>" in decode_tokens:
            #     break




            is_end = prev == args.eod_token
                
    trim_decode_tokens = decode_tokens[:decode_tokens.find("<|endoftext|>")]
    
    return trim_decode_tokens
            

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

from GPT_3.settings import ROOT_DIR

def set_args():
    args=get_args()
    #set up
    #print(args)
    args.deepspeed=True
    args.num_nodes=1
    args.num_gpus=1
    args.model_parallel_size=1
    args.deepspeed_config="script_dir/ds_config.json"
    args.num_layers=32
    args.hidden_size=2560
    #args.load="/home/cognitive/shixw/data/checkpoints/txl-2.8b11-20-15-10"
    # shixw
    args.load = ROOT_DIR + "/data/checkpoints/txl-2.8b11-20-15-10"
    args.num_attention_heads=32
    args.max_position_embeddings=1024
    args.tokenizer_type="ChineseSPTokenizer"
    args.cache_dir="cache"
    args.fp16=True
    args.out_seq_length=700
    args.seq_length=200
    args.mem_length=256
    args.transformer_xl=True
    args.temperature=0.9
    args.top_k=0
    args.top_p=0
    
    return args
def prepare_model(iteration=160000):
    """Main training program."""

    #print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = set_args()
    #print(args)
    args.mem_length = args.seq_length + args.mem_length - 1
 
   
    args.iteration = iteration

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    #get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    #setting default batch size to 1
    args.batch_size = 1
    # iteration

    #generate samples
    return model,tokenizer,args

def generate_strs(strs):
    model,tokenizer,args=prepare_model()
    output=[]
    for str in strs:
        output_string=generate_string(model, tokenizer, args, torch.cuda.current_device(),str)
        print(output_string)
        output.append(output_string)
    return output

def generate():
    fi=[]
    author_list=["楚 屈原","楚 宋玉","汉 孔融","汉 杨修","魏 曹操","魏 曹丕","魏 曹植","魏 王粲","魏 陈琳","吴 诸葛恪","晋 嵇康","晋 潘安","晋 陶渊明","晋 谢灵运","晋 谢安","梁 萧衍","唐 陈子昂","唐 贺知章","唐 王维","唐 骆宾王","唐 李白","唐 杜甫","唐 孟浩然","唐 杜牧","唐 白居易","唐 刘禹锡","唐 柳宗元","南唐 李煜","宋 苏洵","宋 苏轼","宋 欧阳修","宋 曾巩","宋 王安石","宋 李清照","宋 文天祥","宋 苏辙","宋 辛弃疾","宋 岳飞","宋 秦桧","宋 蔡京","明 于谦","明 严世蕃","清 袁枚","清 纪晓岚","清 康有为","清 纳兰性德","清 爱新觉罗弘历"]
    for i in author_list:
        fi.append("咏奥巴马 作者："+i+" 体裁:")
        
    output=generate_strs(fi)

    opt=open("1.txt",'w')
    print(output,file=opt)
    #generate_samples(model, tokenizer, args, torch.cuda.current_device())
    

def qa_common(model, tokenizer, args,content):
    if '问题：' not in content and content[0]!='问':
        content = '问题：' + content
    content = "{} 人工智能 悟能 回答：".format(content)  # ""问题：情侣吵架应该谁先道歉？回答用户：情感专家 回答："
    output_string = generate_string(model, tokenizer, args, torch.cuda.current_device(), content)
    # print("output over : "+content)
    return output_string

#generate()
if __name__ == "__main__":
    import time
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    os.environ["MASTER_PORT"] = "13333"
    content = "程序员如何谈恋爱？&&情感专家&&''"
    model, tokenizer, args = prepare_model()
    b = time.time()
    res = qa_common(model, tokenizer, args,content)
    print(time.time()-b)
    print(res)
