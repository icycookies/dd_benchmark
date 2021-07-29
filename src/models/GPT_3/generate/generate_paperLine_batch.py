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
import nltk
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

import _pickle
import scipy.sparse
import json

def serialize(obj,path,in_json=False,utf8=False):
    if isinstance(obj, np.ndarray):
        np.save(path, obj)
    elif isinstance(obj, scipy.sparse.csr.csr_matrix) or isinstance(obj, scipy.sparse.csc.csc_matrix):
        scipy.sparse.save_npz(path, obj)
    elif in_json:
        if utf8:
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(obj, file, indent=2, ensure_ascii=False)
        else:
            with open(path, 'w') as file:
                json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)


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
            iteration, release, success = get_checkpoint_iteration(args) # add iteration
            iteration = args.iteration
            path = os.path.join(args.load, str(iteration), "mp_rank_00_model_states.pt")
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["module"])
            print(f"Load model file {path}")
        else:
            _ = load_checkpoint(
                model, None, None, args, load_optimizer_states=False)
    # if args.deepspeed:
    #     model = model.module

    return model


def get_batch(context_tokens, device, args): # context_tokens_tensor
    '''
    tokens:         (input_length) -> (batch_size, input_length)
    attention_mask: (1, 1, input_length, args.mem_length+input_length)
    position_ids:   (batch_size, input_length) 
    '''
    tokens = context_tokens
    print("before everything tokens:{}\tsize:{}".format(tokens,tokens.size()))
    # tokens = tokens.view(args.batch_size, -1).contiguous()
    tokens = tokens.unsqueeze(0).repeat(args.batch_size, 1).contiguous() # 复制版
    tokens = tokens.to(device)
    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        transformer_xl=args.transformer_xl,
        mem_length=args.mem_length)

    print("tokens:{}\tsize:{}".format(tokens,tokens.size()))
    print("attention_mask:{}\tsize:{}".format(attention_mask,attention_mask.size()))
    print("position_ids:{}\tsize:{}".format(position_ids,position_ids.size()))

    return tokens, attention_mask, position_ids


def get_list_batch(context_tokens_list, context_length_list, device, args): 
    '''
    tokens:         (batch_size, input_length)
    attention_mask: (1, 1, input_length, args.mem_length+input_length)
    position_ids:   (batch_size, input_length) 
    '''
    org_context_length = max(context_length_list) # 输入文本最大长度 scalar list
    tokens = torch.ones((len(context_tokens_list),org_context_length),dtype=torch.long)*args.eod_token
    for i,context_token in enumerate(context_tokens_list):
        tokens[i,-len(context_token):] = context_token

    tokens = tokens.to(device)
    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        reset_position_ids=False,
        reset_attention_mask=False,
        transformer_xl=args.transformer_xl,
        mem_length=args.mem_length)

    return tokens, attention_mask, position_ids, org_context_length

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.view(1, -1).contiguous()

    return logits

def sample_sequence(model, tokenizer, context_tokens_tensor_list, context_length_list, args, device, mems=None, end_token=None):
# def sample_sequence(model, tokenizer, context_tokens_tensor, context_length, args, device, mems=None, end_token=None):
    counter = 0
    if mems is None:
        mems = []
    if end_token is None:
        end_token = args.eod_token
    already_end = np.zeros(args.batch_size,dtype=np.int32) # 判断各batch是否生成结束, 记录首个endoftext的位置用于输出

    with torch.no_grad():
        tokens, attention_mask, position_ids, org_context_length = get_list_batch(context_tokens_tensor_list, context_length_list, device, args) # context_tokens_tensor_list
        while counter < (args.out_seq_length - org_context_length): 
            index = org_context_length + counter # + 1 # tokens -> beam_tokens[w] 遍历或batch; mems -> *beam_mems[w]
            if counter == 0:
                logits, *mems = model(tokens, position_ids, attention_mask, *mems) # mems: list, length:33, (batch_size, seq_length, hidden_size),[2, 11, 2560]
            else:
                logits, *mems = model(tokens[:, index - 1: index], tokens.new_ones((args.batch_size, 1)) * (index - 1),
                        tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                        dtype=torch.float), *mems) # 这个函数需要看懂 
            logits = logits[:, -1] # 可以在这里加判断，如果出现重复时降低概率重新生成 另外搞向量记录状态
            logits /= args.temperature
            logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
            log_probs = F.softmax(logits, dim=-1) #     log_pbs=torch.log(log_probs) # prev = torch.multinomial(log_probs, num_samples=1)[0] #   prev=torch.multinomial(log_probs,num_samples=beam_size)
            is_end = prev = torch.multinomial(log_probs, num_samples=1) # tensor: (batch_size, num_samples) # prev = prev[0][0] #   prev=torch.multinomial(log_probs,num_samples=beam_size)
            
            mask = is_end.eq(args.eod_token) # is_end.gt(10000) 
            end_positions = torch.nonzero(mask.view(-1), as_tuple=True)[0].cpu().numpy()
            for position in end_positions:
                if already_end[position]==0:
                    already_end[position] = index 

            tokens = torch.cat((tokens, prev), dim=1)# 单个token逐往后续 tokens:(1,generate_length)
            counter += 1

            if np.array(already_end>org_context_length).all():   # 输出 output_tokens_list, _ 
                # print("--------already sentence, break-------")  
                output_tokens_list,  tokens_length_list= list(),list()
                context_tokens_list = [context_token.tolist() for context_token in tokens]
                for i,(start,end) in enumerate(zip(context_length_list,already_end)):
                    raw_decode_tokens = tokenizer.DecodeIds(context_tokens_list[i][org_context_length-start:])
                    trim_decode_tokens = tokenizer.DecodeIds(context_tokens_list[i][org_context_length:end])
                    output_tokens_list.append(trim_decode_tokens)
                    tokens_length_list.append(end-org_context_length)
                return output_tokens_list, tokens_length_list, mems 
        
        if counter >= (args.out_seq_length - org_context_length): # 没有得到结束符但是已经太长了
            # print("--------sentence is long enough-------")  
            already_end[already_end==0] = args.out_seq_length
            output_tokens_list,  tokens_length_list= list(),list()
            context_tokens_list = [context_token.tolist() for context_token in tokens]
            for i,(start,end) in enumerate(zip(context_length_list,already_end)):
                raw_decode_tokens = tokenizer.DecodeIds(context_tokens_list[i][org_context_length-start:])
                trim_decode_tokens = tokenizer.DecodeIds(context_tokens_list[i][org_context_length:end])
                output_tokens_list.append(trim_decode_tokens)
                tokens_length_list.append(end-org_context_length)
            return output_tokens_list, tokens_length_list, mems 

def read_context(tokenizer, args, input_text=''): # 将单句转化为tensor
    terminate_runs, skip_run = 0, 0
    if mpu.get_model_parallel_rank() == 0:
        while True:
            raw_text = input_text  # raw_text = input("\nContext prompt (stop to exit) >>> ")        
            if not raw_text:
                print('Prompt should not be empty!')
                continue
            if raw_text == "stop":
                terminate_runs = 1
                break
            if args.hierarchical:
                raw_text = "Summary: " + raw_text
            context_tokens = tokenizer.EncodeAsIds(raw_text).tokenization # token数 != 词数
            context_length = len(context_tokens)
            if context_length >= args.seq_length: # 长度不符合要求一直输入，这里暂时不用改
                print("\nContext length", context_length,
                      "\nPlease give smaller context than the window length!")
                continue
            break
    else:
        context_length = 0
    terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs]) # xxx_tensor: (1) tensor
    torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    terminate_runs = terminate_runs_tensor[0].item() # terminate_runs:scalar
    if terminate_runs == 1: # 输入stop
        return terminate_runs, raw_text, None, None
    context_length_tensor = torch.cuda.LongTensor([context_length]) # xxx_tensor: (1) tensor
    torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    context_length = context_length_tensor[0].item() # context_length:scalar
    if mpu.get_model_parallel_rank() == 0:
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    else:
        context_tokens_tensor = torch.cuda.LongTensor([0] * context_length)
    torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    return terminate_runs, raw_text, context_tokens_tensor, context_length # scalar, str, 1-dim tensor, scalar


def generate_paperLine(model, tokenizer, args,input_lines,seq_length=256):
    device = torch.cuda.current_device()
    args.seq_length = seq_length
    input_lines = input_lines[:args.batch_size]
    model.eval()
    with torch.no_grad():
        context_tokens_tensor_list, context_length_list, raw_text_list = list(), list(), list()
        torch.distributed.barrier(group=mpu.get_model_parallel_group())
        for input_text in input_lines:
            terminate_runs, raw_text, context_tokens_tensor, context_length = read_context(tokenizer, args, input_text)
            context_tokens_tensor_list.append(context_tokens_tensor)  # tensor list
            context_length_list.append(context_length)  # scalar list max(context_length_list)
            raw_text_list.append(raw_text)
        output_tokens_list, tokens_length_list, _ = sample_sequence(model, tokenizer, context_tokens_tensor_list,context_length_list, args,device )
        if mpu.get_model_parallel_rank() == 0:
            generate_output_list = list()
            for (raw_text, trim_decode_tokens, tokens_length) in zip(raw_text_list, output_tokens_list,tokens_length_list):
                generate_output = dict()
                generate_output['Input'] = raw_text
                generate_output['Output'] = trim_decode_tokens
                generate_output_list.append(generate_output)
                #return generate_output_list
        return generate_output_list



def generate_samples(model, tokenizer, args, device):
    model.eval()
    output_path = "./samples"
    if not os.path.exists(output_path): # output
        os.makedirs(output_path)
    input_path = args.input_path
    with open(input_path,'r',encoding='utf-8') as f_in: # input
        input_lines = f_in.readlines()
        characters = [character.strip() for character in input_lines[0].split(",")] 
        sentences = input_lines[2:]
        print("characters: {}\nsentences: {}\n".format(characters, sentences))
    with torch.no_grad():     
        sentence_counter = 0
        context_tokens_tensor_list, context_length_list, raw_text_list = list(), list(), list()
        print("---------------开始计时:{}---------------".format(datetime.now().strftime('%m-%d-%H-%M-%S')))
        print("args.batch_size:{}".format(args.batch_size))
        for character in characters:
            for sentence in sentences:
                input_text = "问题:" + sentence.strip() + "回答用户:" + character + " 回答:"
                torch.distributed.barrier(group=mpu.get_model_parallel_group())

                # output_text_path = os.path.join(output_path, f"sample-{datetime.now().strftime('%m-%d-%H-%M-%S')}.txt")                
                terminate_runs, raw_text, context_tokens_tensor, context_length = read_context(tokenizer, args, input_text)    # context_length = len(context_tokens)
                # print("Input raw text: {}\nInput length: {}".format(raw_text,len(raw_text)))
                if terminate_runs == 1:
                    return
                context_tokens_tensor_list.append(context_tokens_tensor) # tensor list
                context_length_list.append(context_length) # scalar list max(context_length_list)
                raw_text_list.append(raw_text) # str list
                sentence_counter += 1

                if sentence_counter % args.batch_size == 0: # 等候拼成batch 还可以改为等候时间 这里加判断条件
                    start_time = time.time() # start time
                    output_tokens_list, tokens_length_list, _ = sample_sequence(model, tokenizer, context_tokens_tensor_list, context_length_list, args, device)
                    print("output_tokens_list:{}".format(output_tokens_list))                                    

                    if mpu.get_model_parallel_rank() == 0:   
                        output_json_path = os.path.join(output_path, f"sample-{datetime.now().strftime('%m-%d-%H-%M-%S')}.json")
                        generate_output_list = list()     
                        for (raw_text,trim_decode_tokens,tokens_length) in zip(raw_text_list,output_tokens_list,tokens_length_list):
                            # output_json_path = os.path.join(output_path, f"sample-{datetime.now().strftime('%m-%d-%H-%M-%S')}.json")
                            generate_output = dict()
                            generate_output['Time'] = time.time() - start_time
                            generate_output['Input'] = raw_text
                            generate_output['Output'] = trim_decode_tokens 
                            generate_output['Input_length'] = max(context_length_list)#len(raw_text)
                            generate_output['Output_length'] = str(tokens_length) 
                            generate_output_list.append(generate_output)
                        print("generate_output_list:{}".format(generate_output_list))
                        serialize(generate_output_list,output_json_path,True,True) # temporarily alter
                    
                    context_tokens_tensor_list, context_length_list, raw_text_list = [], [], []
                    sentence_counter = 0

                    torch.distributed.barrier(group=mpu.get_model_parallel_group())

        # 最后未形成batch的部分        
        if len(raw_text_list)>0: 
            start_time = time.time() # start time
            output_tokens_list, tokens_length_list, _ = sample_sequence(model, tokenizer, context_tokens_tensor_list, context_length_list, args, device)
            generate_output_list = list()     
            print("output_tokens_list:{}".format(output_tokens_list))                                    
            if mpu.get_model_parallel_rank() == 0:        
                for (raw_text,trim_decode_tokens,tokens_length) in zip(raw_text_list,output_tokens_list,tokens_length_list):
                    # output_json_path = os.path.join(output_path, f"sample-{datetime.now().strftime('%m-%d-%H-%M-%S')}.json")
                    generate_output = dict()
                    generate_output['Time'] = time.time() - start_time
                    generate_output['Input'] = raw_text
                    generate_output['Output'] = trim_decode_tokens 
                    generate_output['Input_length'] = max(context_length_list)#len(raw_text)
                    generate_output['Output_length'] = str(tokens_length) 
                    generate_output_list.append(generate_output)
                print("generate_output_list:{}".format(generate_output_list))
                serialize(generate_output_list,output_json_path,True,True) # temporarily alter
        print("---------------结束计时:{}---------------".format(datetime.now().strftime('%m-%d-%H-%M-%S')))



def prepare_tokenizer(args):
    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir,
        'add_eop': args.hierarchical}
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
    args.eod_token = tokenizer.get_command('eos').Id # 'eos'+'pad'


    # after = tokenizer.num_tokens
    # while after % mpu.get_model_parallel_world_size() != 0:
    #     after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer


def set_args(args):
    args.iteration=160000 # 这里指定调用的模型
    args.input_path="./test_data/test_data_2.txt" # 这里指定输入文件
    args.inference_batch_size=10  # 这里指定batch_size
    args.deepspeed=True
    args.num_nodes=1
    args.num_gpus=1
    args.model_parallel_size=1 # 
    args.deepspeed_config="script_dir/ds_config.json"
    args.num_layers=32
    args.hidden_size=2560
    args.load="/home/yjf/shixw/shi_BlockLM-master/data/checkpoints/txl-2.8b11-20-15-10"
    args.hierarchical = False
    args.num_attention_heads=32
    args.max_position_embeddings=1024
    args.tokenizer_type="ChineseSPTokenizer"
    args.cache_dir="cache"
    args.fp16=True
    args.out_seq_length=512 # 这里指定最大文本长度
    args.seq_length=512
    args.mem_length=256
    args.transformer_xl=True
    args.temperature=0.9
    args.top_k=0
    args.top_p=0

    return args

def prepare_model():
    """Main training program."""

    print('Generate Samples')

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args() # 
    args.mem_length = args.seq_length + args.mem_length - 1 # seq_length/mem_length: 512+256, seq-length: context_length
    args = set_args(args)

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    # setting default batch size to 1 # batch_size改这里
    # args.batch_size = 2
    args.batch_size = args.inference_batch_size

    return model, tokenizer, args


if __name__ == "__main__":
    input_lines = ["杭州的你，择偶标准是什么样的？","为什么现在男生不主动但是还总抱怨没对象？","男朋友买鸭脖自己全吃了没给我，应该分手吗？"]
    model, tokenizer, args = prepare_model()
    out_list = generate_paperLine(model, tokenizer, args,input_lines)
    print(out_list)



