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
            iteration=160000 # 104000
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
def checklength(s):
    w=s.replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace('<','').replace(' ','').replace('>','').replace('《','').replace('》','').replace('\"','').replace('“','').replace('”','').replace("‘",'').replace('‘','').replace('、','').replace('-','').replace('.','')
    return len(w)
def checkpoem(s,dks):  #check the score of poem
    w=s.replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace('<','').replace(' ','').replace('>','').replace('《','').replace('》','').replace('\"','').replace('“','').replace('”','').replace("‘",'').replace('‘','').replace('、','').replace('-','')
    
    if '[' in w:
        return 0
    if '【' in w:
        return 0
    if '(' in w:
        return 0
    if ')' in w:
        return 0
    if ':' in w:
        return 0
    if '：' in w:
        return 0
    if '⁇' in w:
        return 0
    if '.' in w:
        return 0
    for i in range(10):
        if str(i) in w:
            return 0
    sentence=w.split(',')
    if (dks==1) and((len(sentence)<4) or (len(sentence)%2==1)):
        return 0
    
    if len(sentence[-1])>7:
        return 0
    if len(sentence)>1:
        lengthofpoem=len(sentence[0])
        if not(lengthofpoem in [4,5,7]):
            return 0
        for i in range(len(sentence)):
            if (len(sentence[i])!=lengthofpoem) and (i!=len(sentence)-1 or dks==1):
                return 0
            if i>=1:
                if len(sentence[i])>len(sentence[i-1]):
                    return 0
                e2=0
                for j in range(len(sentence[i])):
                    if sentence[i][j]==sentence[i-1][j]:
                        return 0
                    if (i>=2) and (sentence[i][j]==sentence[i-2][j]):
                        e2+=1
                if e2>=2:
                    return 0
                for j in range(i):
                    if sentence[i]==sentence[j]:
                        return 0
    #移除合掌，
    if len(sentence)>=16 and len(sentence[-1])==lengthofpoem:
        return 2
    return 1

def generate_string(model, tokenizer, args, device,input_str):
    
    context_count=0
    model.eval()
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
        eo_tokens=tokenizer.EncodeAsIds('<|endoftext|>').tokenization
        context_length = len(context_tokens)
        if context_length>=args.seq_length:
            return "输入过长。"
      
        #terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
                # pad_id = tokenizer.get_command('pad').Id
                # if context_length < args.out_seq_length:
                #     context_tokens.extend([pad_id] * (args.out_seq_length - context_length))

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        eo_token_tensor=torch.cuda.LongTensor(eo_tokens)
        context_length_tensor = torch.cuda.LongTensor([context_length])
        context_length = context_length_tensor[0].item()
        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)

        start_time = time.time()

        counter, mems = 0, []
        org_context_length = context_length
        beam_size=20
        beam_candidate=10
        beam_max=3
        final_storage=[]
        final_storage_score=[]
        while counter < (args.out_seq_length - org_context_length):
            if counter==0:
                hasended=np.zeros(beam_size)
                beam_tokens=[]
                beam_score=[]
                beam_mems=[]
                logits,*mems = model(tokens, position_ids, attention_mask, *mems)
                logits = logits[:, -1]
                logits /= args.temperature
                #logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
                log_probs = F.softmax(logits, dim=-1)
                log_pbs=torch.log(log_probs)
                    #print(log_probs)
                prev=torch.multinomial(log_probs,num_samples=beam_size)
                for i in range(beam_size):
                    #new_tokens.append(prev[0,i].view(1,1))
                    beam_score.append(log_probs.data[0,prev[0,i]].cpu().numpy())
                    beam_mems.append(mems)
                    beam_tokens.append(torch.cat((tokens,prev[0,i].view(1,1)),dim=1))
                #print(counter,beam_tokens,beam_score)
            else:
                beam_score_current=[]
                beam_score_ids=[]
                new_beam_mems=[]
                beam_token=[]
                
                for w in range(beam_size):
                    index = org_context_length + counter
                    logits, *mems = model(beam_tokens[w][:, index - 1: index], beam_tokens[w].new_ones((1, 1)) * (index - 1),
                    beam_tokens[w].new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                              dtype=torch.float), *beam_mems[w])
                    logits = logits[:, -1]
                    logits /= args.temperature
                    #logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
                    log_probs = F.softmax(logits, dim=-1)
                    log_pbs=torch.log(log_probs)
                    #print(log_probs)
                    
                    prev = torch.multinomial(log_probs,num_samples=beam_candidate)
                    
                    for i in range(beam_candidate):
                        beam_score_current.append(beam_score[w]+log_pbs.data[0,prev[0,i]].cpu().numpy())
                        beam_score_ids.append([w,i])
                        beam_token.append(prev[0,i])
                    new_beam_mems.append(mems)
                    
                    
                gt=np.argsort(beam_score_current)
                
                import random
                if counter<150:
                    random.shuffle(gt)
                
                beam_score_new=[]
                beam_token_new=[]
                w=1
                cur=0
                if beam_size>0:
                    beam_size=20
                    beamsum=np.zeros(beam_size)
                
                while cur<beam_size:
                    sur=beam_score_ids[gt[-w]]
                    beam_token_potential=torch.cat((beam_tokens[sur[0]],beam_token[gt[-w]].view(1,1)),dim=1)
                    decode_tokens=tokenizer.DecodeIds(beam_token_potential.view(-1).contiguous().tolist())
                    dks=0
                    if "<|endoftext|>" in decode_tokens:
                        dks=1
                    trim_decode_tokens=decode_tokens[len(input_str):decode_tokens.find("<|endoftext|>")]
                    #print(trim_decode_tokens)
                    poem=checkpoem(trim_decode_tokens,dks)
                    #print(w,trim_decode_tokens)
                    beam_score_penalty=0
                    if poem==2:
                        beam_token_potential=torch.cat((beam_token_potential,eo_token_tensor.view(1,7)),dim=1)
                        beam_score_penalty=-15
                        poem=1
                        dks=1
                        #print(beam_token_potential)
                        
                    if (poem==1 and beamsum[sur[0]]<beam_max):
                        if dks==0:
                            beamsum[sur[0]]+=1
                            beam_score_new.append(beam_score_current[gt[-w]]+beam_score_penalty)
                            beam_token_new.append(beam_token_potential)
                            beam_mems[cur]=new_beam_mems[sur[0]]
                            cur+=1
                        if dks==1:
                            lengthtk=checklength(trim_decode_tokens)
                            lengthtok=len(beam_token_potential)-15
                            final_storage.append(trim_decode_tokens)
                            final_storage_score.append((beam_score_current[gt[-w]]+beam_score_penalty)/lengthtk**0.4/lengthtok**0.4)
                    w+=1
                    if w==len(gt):
                        beam_size=cur
                  
                    
                beam_score=beam_score_new
                beam_tokens=beam_token_new
                #tokens = torch.cat((tokens, prev.view(1, 1)), dim=1)
            context_length += 1
            counter += 1
            #print(counter,beam_token[0])
        #print(final_storage,final_storage_score)
        if len(final_storage_score)==0:
            return input_str+"才思枯竭，写不出来。"
    
        i=np.argmax(final_storage_score)
        return input_str+final_storage[i]
        
            

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
from GPT_3.settings import GENERATE_DIR
from GPT_3.settings import ROOT_DIR
#import timeout_decorator

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
    #args.load="data/checkpoints/txl-2.8b11-20-15-10"
    args.load=ROOT_DIR + "/data/checkpoints/txl-2.8b11-20-15-10"
    args.num_attention_heads=32
    args.max_position_embeddings=1024
    args.tokenizer_type="ChineseSPTokenizer"
    args.cache_dir="cache"
    args.fp16=True
    args.out_seq_length=220
    args.seq_length=200
    args.mem_length=256
    args.transformer_xl=True
    args.temperature=1.0
    args.top_k=0
    args.top_p=0
    
    return args
def prepare_model():
    """Main training program."""
    #os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    #print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = set_args()
    #print(args)
    args.mem_length = args.seq_length + args.mem_length - 1
    

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
    title="赠丁铭"
    # author_list=["民国 张宗昌","楚 屈原","楚 宋玉","汉 孔融","汉 杨修","魏 曹操","魏 曹丕","魏 曹植","魏 王粲","魏 陈琳","吴 诸葛恪","晋 嵇康","晋 潘安","晋 陶渊明","晋 谢灵运","晋 谢安","梁 萧衍","唐 陈子昂","唐 贺知章","唐 王维","唐 骆宾王","唐 李白","唐 杜甫","唐 孟浩然","唐 杜牧","唐 白居易","唐 刘禹锡","唐 柳宗元","南唐 李煜","宋 苏洵","宋 苏轼","宋 欧阳修","宋 曾巩","宋 王安石","宋 李清照","宋 文天祥","宋 苏辙","宋 辛弃疾","宋 岳飞","宋 秦桧","宋 蔡京","明 于谦","明 严世蕃","清 袁枚","清 纪晓岚","清 康有为","清 纳兰性德","清 爱新觉罗弘历"]
    author_list=["民国 张宗昌"]
    for i in author_list:
        fi.append(title+" 作者:"+i+" 体裁:诗歌 题名:"+title+" 正文:")
        
    output=generate_strs(fi)

    opt=open("1.txt",'w')
    print(output,file=opt)
    #generate_samples(model, tokenizer, args, torch.cuda.current_device())
    

#generate()
#generate()
#@timeout_decorator.timeout(300)
def write_common(content,model, tokenizer, args):
    print("-------------")
    output_string = generate_string(model, tokenizer, args, torch.cuda.current_device(), content)
    print("output over : "+content)
    return output_string

