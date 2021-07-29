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
import copy
from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0
from pretrain_gpt2 import get_model
from pypinyin import pinyin,FINALS, FINALS_TONE,TONE3
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
            # shixw
            iteration=104000
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

def generate_score(model, tokenizer, args, device, input_str, eval_str):
    #penalty on same word
    penalty=0
    for i in eval_str:
        if i in input_str[:-7]:
            penalty+=1
    context_count = 0
    model.eval()
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
        eval_tokens = tokenizer.EncodeAsIds(eval_str).tokenization + [tokenizer.get_command('eos').Id]
        if len(context_tokens)==0:
            context_tokens = eval_tokens[0:1]
            eval_tokens = eval_tokens[1:]
        context_length = len(context_tokens)
        eval_length = len(eval_tokens)
        if context_length >= args.seq_length:
            return "输入过长。"

        # terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
        # pad_id = tokenizer.get_command('pad').Id
        # if context_length < args.out_seq_length:
        #     context_tokens.extend([pad_id] * (args.out_seq_length - context_length))

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        eval_tokens_tensor = torch.cuda.LongTensor([eval_tokens])
        context_length_tensor = torch.cuda.LongTensor([context_length])
        eval_length_tensor = torch.cuda.LongTensor([eval_length])
        # context_length = context_length_tensor[0].item()
        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)
        # print(context_tokens)
        start_time = time.time()

        counter, mems = 0, []
        org_context_length = context_length
        sumlognum = 0
        while counter < eval_length:
            if counter == 0:
                logits, *mems = model(tokens, position_ids, attention_mask, *mems)
                logits = logits[:, -1]
            else:
                index = org_context_length + counter
                logits, *mems = model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                                      tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                      dtype=torch.float), *mems)
                logits = logits[:, 0]
            # logits = logits[:, -1]
            #logits /= args.temperature
           # logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
            log_probs = F.softmax(logits, dim=-1)
            log_num = torch.log(log_probs).data
            # print(log_num)
            sumlognum += log_num[0, eval_tokens[counter]]
            # print(log_probs)
            # prev = torch.multinomial(log_probs, num_samples=1)[0]
            # print(tokens,eval_tokens_tensor[counter:counter+1])
            tokens = torch.cat((tokens, eval_tokens_tensor[:, counter:counter + 1]), dim=1)
            # print(tokens,sumlognum)
            context_length += 1
            counter += 1

        # trim_decode_tokens = decode_tokens[:decode_tokens.find("<|endoftext|>")]
        sumlognum = sumlognum
        del logits
        del mems
        torch.cuda.empty_cache()
        return sumlognum-2.5*(penalty**2.5)
        
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

def generate_token_tensor(str,tokenizer):
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(str).tokenization
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        return context_tokens_tensor
    
def checksentence(sentence,original_context,min_length,max_length,endnote):
    if "<|end" in sentence:
        return 0
            
    if ((len(sentence)>max_length and not(sentence[-1] in endnote)) or len(sentence)==0) or len(sentence)>max_length+1:
        return 1
    if (sentence[-1] in endnote)and ((len(sentence)<=min_length) or (len(sentence)==7)):
        return 1
            
    if (sentence[-1] in endnote)and (sentence[:-1] in original_context):
        return 1
    last=getlastsentence(original_context)
    mdisobey=0
    illegal_notes=[' ',':','《','》','‘','“','-','——','⁇','[','【','】',']','.','、','(','（',')','）','·']
    for i in range(10):
        illegal_notes.append(str(i))
    for i in range(64,123):
        illegal_notes.append(chr(i))
    for note in illegal_notes:
        if note in sentence:
            return 1
    if min_length==max_length:
        l1=pinyin(sentence,style=TONE3)
        l2=pinyin(last,style=TONE3)
        #print(l1,l2)
        disobey=0
        if len(l1)!=len(sentence):
            return 1
        for i in range(len(sentence)):
            if (not(sentence[i] in endnote)) and (i<len(l1)) and (i<len(l2)):
                st=l1[i][0][-1]
                if st in ['1','2']:
                    st1=1
                else:
                    st1=2
                sr=l2[i][0][-1]
                if sr in ['1','2']:
                    sr1=1
                else:
                    sr1=2
                if st1+sr1!=3:
                    disobey+=0.35
                    if i%2==1:
                        disobey+=0.35
                    if i==max_length-1:
                        disobey+=0.6
        if ((',' in last) or ('，' in last)):
            disobey*=2
        if disobey>max_length*0.5:
            return 1
        
        
    for i in range(len(sentence)):
       # if sentence[i]=="柯":
        #    print(sentence[i],last[i],sentence[i]==last[i])
        if min_length==max_length:
            if (i<len(last)-1) and (sentence[i]==last[i]):
                #print(sentence,last)
                return 1
                
            
        
        if i<len(sentence)-3:
            if sentence[i:i+3] in original_context:
                return 1
    
    if (sentence[-1] in endnote):
        return 0
        
        
    return 2
    
             

    
def generate_sentence(model,tokenizer,args,device,current_tokens,mems,endnote=[",","，","?","？"],num_candidates=10,min_length=4,max_length=7):
    model.eval()
    with torch.no_grad():
        #index=len(tokens[0])
        mct_tree=[]
        if min_length!=max_length:
            mems=[]
            tokens, attention_mask, position_ids = get_batch(current_tokens, device, args)
            logits,*rts = model(tokens, position_ids, attention_mask, *mems)
        else:
            tokens=current_tokens
            index=len(tokens[0])
            logits,*rts=model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                        tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                            dtype=torch.float), *mems)
                                                            
        output_tokens_list = tokens.view(-1).contiguous()
        original_context=tokenizer.DecodeIds(output_tokens_list.tolist())
        context_length=len(tokens[0])
        logits=logits[0,-1]
        #mct_structure=-np.ones(len(logits))
        mct_tree.append([logits,rts,tokens,-np.ones(len(logits)),torch.ones(len(logits)).cuda(),0])
        #print(logits.shape)
        final_result=[]
        nextid=0
        tries=0
        max_tries=num_candidates*30
        while (len(final_result)<num_candidates)and(tries<max_tries):
            currentid=nextid
            tries+=1
            while currentid!=-1:
                logits=mct_tree[currentid][0].view(-1)-torch.log(mct_tree[currentid][4])*0.5
                log_probs = F.softmax(logits/args.temperature, dim=-1)
              
                pr=torch.multinomial(log_probs,num_samples=1)[0]
                #pr=torch.argmax(logits)
                prev=pr.item()
                #print(logits.shape,currentid,prev)
                mct_tree[currentid][4][prev]+=1
                lastid=currentid
                currentid=int(mct_tree[currentid][3][prev])
            #start from lastid & currentid
            
            cqs=mct_tree[lastid][2]
            #print(pr)
            tokens = torch.cat((cqs, pr.unsqueeze(0).view(1, 1)), dim=1)
            output_tokens_list = tokens.view(-1).contiguous()
            #if max_length==min_length:
             #   print(min_length,output_tokens_list,context_length)
            sentence = tokenizer.DecodeIds(output_tokens_list[context_length:].tolist())
            
            #print(output_tokens_list[context_length:],context_length,sentence)
            logit=mct_tree[lastid][0]
            log_probs = F.softmax(logit, dim=-1)
            log_pbs=torch.log(log_probs)
            score=log_pbs[prev].item()
            nextid=0
            ip=checksentence(sentence,original_context,min_length,max_length,endnote)
       
            if (ip==1):
                mct_tree[lastid][0][prev]=-10000
                continue
            if (ip==0):
                mct_tree[lastid][0][prev]=-10000
                final_result.append([sentence,mct_tree[lastid][5],copy.deepcopy(tokens),copy.deepcopy(mct_tree[lastid][1])])
                #print(sentence,score)
                continue
        
           
            
                #calculate
            mct_tree[lastid][3][prev]=len(mct_tree)
            rts=mct_tree[lastid][1]
            index=len(tokens[0])
            
            
            logits,*rts=model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                        tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                                dtype=torch.float), *rts)
            logits=logits[0,-1]
            score=mct_tree[lastid][5]+score
            mct_tree.append([logits,rts,tokens,-np.ones(len(logits)),torch.ones(len(logits)).cuda(),score])
            nextid=len(mct_tree)-1
        del mct_tree
        torch.cuda.empty_cache()
        #print(tries,len(final_result))
        return final_result
def getlength(str):
    w=str.replace('。',',').replace('，',',').replace('？',',').replace('?',',')
    sp=w.split(',')
    
    return len(sp[-2])

def getlastsentence(str):
    w=str.replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace(' ',',').replace('！',',').replace('!',',').replace(':',',').replace(' ','')
    sp=w.split(',')
    fom=sp[-1]
    if len(fom)==0:
        fom=sp[-2]
    return fom+w[-1]

def get2sentencebefore(str):
    w=str.replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace(' ',',').replace('！',',').replace('!',',').replace(':',',').replace(' ','')
    sp=w.split(',')
    
    return sp[-3]

def check2compare(sentence1,sentence2,imp):
    s1=sentence1.replace('。','').replace('，','').replace('？','').replace('?','').replace(' ','').replace('！','').replace('!','').replace(',','')
    s2=sentence2.replace('。','').replace('，','').replace('？','').replace('?','').replace(' ','').replace('！','').replace('!','').replace(',','')
    if len(s1)!=len(s2):
        return -1000
    num=0
    for i in range(len(s1)):
        if s1[i]==s2[i]:
           num+=1
        
    score=0.5-num*num*2
           
    w1=pinyin(s1,style=FINALS)[-1]
    w2=pinyin(s2,style=FINALS)[-1]
    if (w1!=w2) or (s1[-1]==s2[-1]):
        score-=imp*0.6
    else:
        score+=imp
    return score
    
    
    
    
def generate_string(model, tokenizer, args, device,title,author):
    input_str=title+" 作者:"+author+" 体裁:诗歌 题名:"+title+" 正文: "
    # aus=author.split(' ')[1]
    input_len=len(input_str)
    context_count=0
    model.eval()
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
        eo_tokens=tokenizer.EncodeAsIds('<|endoftext|>').tokenization
        context_length = len(context_tokens)
        if context_length>=args.seq_length:
            return "输入过长。"
      

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        eo_token_tensor=torch.cuda.LongTensor(eo_tokens)
        context_length_tensor = torch.cuda.LongTensor([context_length])
        context_length = context_length_tensor[0].item()
        #tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)

        start_time = time.time()

        counter, mems = 0, []
        org_context_length = context_length
        beam_size=10
        # shixw
        beam_candidate=2
        beam_max=2
        final_storage=[]
        final_storage_score=[]
        step=16
        overall_score=[]
        past_beam_id=[]
        #print(counter,beam_tokens,beam_score)
        # beam_sentences=generate_sentence(model,tokenizer,args,device,context_tokens_tensor,[],num_candidates=beam_size*5)
        # shixw
        beam_sentences=generate_sentence(model,tokenizer,args,device,context_tokens_tensor,[],num_candidates=beam_size*1)
        for w in range(len(beam_sentences)):
            if '<|end' in beam_sentences[w][0]:
                continue
            input='”'+beam_sentences[w][0]+'”此句出自诗歌《'
            output_str=title+'》'
            score1=generate_score(model,tokenizer,args,device,input,output_str)
            '''
            input='”'+beam_sentences[w][0]+'”此句作者为'
            output_str=aus
            score2=generate_score(model,tokenizer,args,device,input,output_str)
            '''
            ss=-beam_sentences[w][1]/len(beam_sentences[w][0])-7
            iscore=score1-0.5*(np.abs(ss)+ss)
            beam_sentences[w][1]=iscore
            #print(beam_sentences[w][0],beam_sentences[w][1])
            overall_score.append(iscore)
            past_beam_id.append(w)
            
        gy=np.argsort(overall_score)
        k=0
        sumbeam=np.zeros(100)
        gym=[]
        num=0
        while num<beam_size:
           k+=1
           if sumbeam[past_beam_id[gy[-k]]]<beam_max:
            sumbeam[past_beam_id[gy[-k]]]+=1
            gym.append(gy[-k])
            num+=1
        best_score=-1000
        best_pos=0
        for i in range(step):
            if (best_score>-1000) and (i>8):
                return final_storage[best_pos]
            beam_new_sentences=[]
            
            endnote=[',','，','?','？']
            if i%2==0:
                endnote=['。','?','？','！','!']
            overall_score=[]
            past_beam_id=[]
            size=beam_size
            if len(gym)<size:
                size=len(gym)
            ini_score=beam_sentences[gym[0]][1]/(i+1)
            # early stopping
            if i>7:
                ini_score-=0.6
            if i>11:
                ini_score-=1.4
            if ini_score<best_score-2:
                print(ini_score,best_score,best_pos)
                return final_storage[best_pos]
            
            for w in range(size):
                id=gym[w]
                current_sentence=input_str+beam_sentences[id][0]
               #print(beam_sentences[id][0],beam_sentences[id][1])
                ini_score=beam_sentences[id][1]
                token_tensor=beam_sentences[id][2]
                mems=beam_sentences[id][3]
            
                len_sentence=getlength(beam_sentences[id][0])
                if i>=15:
                    final_storage.append(current_sentence[input_len:])
                    sc=beam_sentences[id][1]/(i+1)
                    sc-=2
                    final_storage_score.append(sc)
                    print(current_sentence,final_storage_score[-1])
                    continue
                #print(token_tensor)
                gen=generate_sentence(model,tokenizer,args,device,token_tensor,mems,num_candidates=beam_candidate,endnote=endnote,min_length=len_sentence,max_length=len_sentence)
                for jj in gen:
                    if '<|end' in jj[0]:
                        if (i%2==1 and i>=3):
                            final_storage.append(current_sentence[input_len:])
                            sc=beam_sentences[id][1]/(i+1) #prioritize short poems
                            if (i==5 or i==9 or i==13):
                                sc-=2
                            if (i==11):
                                sc-=0.6
                            if (i==3):
                                sc+=0.6
                            final_storage_score.append(sc)
                            print(current_sentence,final_storage_score[-1])
                            if sc>best_score:
                                best_score=sc
                                best_pos=len(final_storage)-1
                        continue
                    st=jj[0]
                    if i%2==1:
                        st=getlastsentence(beam_sentences[id][0])+jj[0]
                    input='”'+st+'”此句出自诗歌《'
                    
                    output_str=title+'》'
                    
                    score1=generate_score(model,tokenizer,args,device,input,output_str)
                    '''
                    input='”'+st+'”此句作者为'
                    output_str=aus
                    score2=generate_score(model,tokenizer,args,device,input,output_str)
                    '''
                    
                    ss=-jj[1]/len(jj[0])-7
                    iscore=score1-(np.abs(ss)+ss)*0.5
                    if i>=1:
                        before=get2sentencebefore(beam_sentences[id][0])
                        imp=1
                        if i%2==0:
                            imp+=1.5
                        #print(i,beam_sentences[id][0],before,jj[0])
                        scorem=check2compare(before,jj[0],imp)
                        iscore+=scorem
                        
                        
                    jj[0]=beam_sentences[id][0]+jj[0]
                    jj[1]=iscore+ini_score
                    #print(jj[0],jj[1])
                    beam_new_sentences.append(jj)
                    overall_score.append(jj[1])
                    past_beam_id.append(w)
            del beam_sentences
            torch.cuda.empty_cache()
            beam_sentences=beam_new_sentences
            gy=np.argsort(overall_score)
            sumbeam=np.zeros(100)
            k=0
            gym=[]
            num=0
            while (num<beam_size) and (k+1<len(past_beam_id)):
                k+=1
                
                if sumbeam[past_beam_id[gy[-k]]]<beam_max:
                    sumbeam[past_beam_id[gy[-k]]]+=1
                    gym.append(gy[-k])
                    num+=1
                
            
        if len(final_storage)>0:
            return "太难了，写不出来。"
        
        max_score=np.argmax(final_storage_score)
        
        del beam_sentences
        del beam_new_sentences
        torch.cuda.empty_cache()
        
        return final_storage[max_score]
        
            

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
# shixw
from GPT_3.settings import ROOT_DIR
def set_args():
    args=get_args()
    #set up
    #print(args)
    args.deepspeed=True
    args.num_nodes=1
    args.num_gpus=1
    args.model_parallel_size=1
    # shixw
    args.load= ROOT_DIR + "/data/checkpoints/txl-2.8b11-20-15-10"
    # args.load = "/home/cognitive/shixw/data/checkpoints/txl-2.8b11-20-15-10"
    args.num_layers=32
    args.hidden_size=2560
    args.num_attention_heads=32
    args.max_position_embeddings=1024
    args.tokenizer_type="ChineseSPTokenizer"
    args.cache_dir="cache"
    args.fp16=True
    args.out_seq_length=180
    args.seq_length=200
    args.mem_length=256
    args.transformer_xl=True
    args.temperature=0.96
    args.top_k=0
    args.top_p=0
    
    return args
# shixw
# def set_gpu():
#      available_gpu_list = os.environ.get('AVAILABLE_GPU_LIST')
#      this_gpu_index = available_gpu_list.pop()

def prepare_model():
    """Main training program."""

    #print('Generate Samples')
    # shixw
    # os.environ["MASTER_PORT"] = "12200"

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

def generate_strs(tups):
    model,tokenizer,args=prepare_model()
    output=[]
    for tup in tups:
        #str=generate_token_tensor(str,tokenizer)
        output_string=generate_string(model, tokenizer, args, torch.cuda.current_device(),tup[0],tup[1])
        print(output_string)
        output.append(output_string)
    return output

    
    
def generate():
    fi=[]
    #title_list=["咏希拉里","咏拜登","咏蔡英文","早春","登黄鹤楼","上甘岭","赠抗疫英雄"]
    title_list=["春花","夏风","秋月","冬雪","桃花潭送刘知远大师"]
    title_list=["登勃朗峰","登特朗普大厦","登艾菲尔铁塔","咏印度","咏北条时宗"]
    author_list=["民国 张宗昌","唐 杜甫","唐 李白","清 曹雪芹","唐 温庭筠","当代 施政彤","唐 贺知章","唐 李商隐","唐 孟浩然","宋 欧阳修","宋 曾巩","宋 王安石","楚 屈原","楚 宋玉","汉 孔融","汉 杨修","魏 曹操","魏 曹丕","魏 曹植","魏 王粲","魏 陈琳","吴 诸葛恪","晋 嵇康","晋 潘安","晋 陶渊明","晋 谢灵运","晋 谢安","梁 萧衍","唐 陈子昂","唐 王维","唐 骆宾王","唐 杜牧","唐 白居易","唐 刘禹锡","唐 柳宗元","南唐 李煜","宋 苏洵","宋 苏轼","宋 李清照","宋 文天祥","宋 苏辙","宋 辛弃疾","宋 岳飞","宋 秦桧","宋 蔡京","明 于谦","明 严世蕃","清 袁枚","清 纪晓岚","清 康有为","清 纳兰性德","清 爱新觉罗弘历"]
    for i in title_list:
        fi.append([i,author_list[0]])
        
    output=generate_strs(fi)

    opt=open("1.txt",'w')
    print(output,file=opt)
    #generate_samples(model, tokenizer, args, torch.cuda.current_device())
    

# generate()
# shixw
def write_common(content,model, tokenizer, args):
    title = content.split(" ",1)[0]
    author = content.split(" ",1)[1]
    output_string = generate_string(model, tokenizer, args, torch.cuda.current_device(),title,author)
    print("output over : "+title + " " + author)
    return output_string


if __name__ == "__main__":
    os.environ["MASTER_PORT"] = "12021"
    model, tokenizer, args = prepare_model()
    print(write_common("登勃朗峰 现代 文哥",model, tokenizer, args))


