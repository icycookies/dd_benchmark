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
open_old_pronounce=1
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
from com_utils.http_utils import CanNotReturnException, InputTooLongException
import jsonlines
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

def generate_score_past(model, tokenizer, args, device, input_str, eval_str):
    #penalty on same word
    penalty=0
    for i in eval_str:
        if i in input_str[:-4]:
            penalty+=1
    context_count = 0
    model.eval()
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
        eval_tokens = tokenizer.EncodeAsIds(eval_str).tokenization
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
 
def generate_score(model, tokenizer, args, device,  mid_str, eval_str, raw_mems=None):
    penalty=0
    title=mid_str[:-4]
    for i in eval_str:
        if i in title:
            penalty+=1
    context_count = 0
    model.eval()
    
   
    mems = []
   

    def build_mask_matrix(query_length, key_length, sep=0, device='cuda'):
        m = torch.ones((1, query_length, key_length), device=device)
        assert query_length <= key_length
        m[0, :, -query_length:] = torch.tril(m[0, :, -query_length:])
        m[0, :, :sep + (key_length - query_length)] = 1
        m = m.unsqueeze(1)
        return m
    #penalty on same word

    model.eval()
    with torch.no_grad():
               
        mid_tokens = tokenizer.EncodeAsIds(mid_str).tokenization
        eval_tokens = tokenizer.EncodeAsIds(eval_str).tokenization
        context_tokens=mid_tokens

        context_length = len(context_tokens)
        eval_length = len(eval_tokens)

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens  + eval_tokens)

        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)
        # print(context_tokens)
        start_time = time.time()

        index = 0
        logits, *mems = model(tokens[:, index: ],
            torch.arange(index, tokens.shape[1], dtype=torch.long, device=tokens.device).unsqueeze(0),
            build_mask_matrix(tokens.shape[1] - index, tokens.shape[1], device=tokens.device),
            *mems)
            
        logits = logits[:, -eval_length-1:-1]
        log_probs = F.softmax(logits, dim=-1)
        log_num = torch.log(log_probs).data.clamp(min=-35, max=100000)

        log_nums = [
            log_num[0, i, eval_token]
            for i, eval_token in enumerate(eval_tokens) # TODO eos
        ]
        #print(log_nums)

        sumlognum = sum(log_nums)
           
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
    w=s.replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace('<','').replace(' ','').replace('>','').replace('《','').replace('》','').replace('\"','').replace('“','').replace('”','').replace("‘",'').replace('‘','').replace('、','').replace('-','').replace('.','').replace('!',',').replace('！',',')
    return len(w)

def generate_token_tensor(str,tokenizer):
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(str).tokenization
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        return context_tokens_tensor

rus=set(['八','搭','塌','邋','插','察','杀','煞','夹','俠','瞎','辖','狹','匣','黠','鸭','押','压','刷','刮','滑','猾','挖','蜇','舌','鸽','割','胳','搁','瞌','喝','合','盒','盍','曷','貉','涸','劾','核','钵','剝','泼','摸','脱','托','捋','撮','缩','豁','活','切','噎','汁','织','隻','掷','湿','虱','失','十','什','拾','实','食','蝕','识','石','劈','霹','滴','踢','剔','屐','积','激','击','漆','吸','息','媳','昔','席','锡','檄','觋','揖','一','壹','扑','匍','仆','弗','紱','拂','福','蝠','幅','辐','服','伏','茯','督','突','秃','俗','出','蜀','窟','哭','忽','惚','斛','鹄','屋','屈','诎','曲','戌','拍','塞','摘','拆','黑','勺','芍','嚼','粥','妯','熟','白','柏','伯','薄','剥','摸','粥','轴','舳','妯','熟','角','削','学'])
ss=set(['de','te','le','ze','ce','se','fa','fo','dei','zei','gei','hei','sei','bie','pie','mie','die','tie','nie','lie','kuo','zhuo','chuo','shuo','ruo'])
def checkpz(st,wd):
    #入声字判断
    
    #轻声按失败算。
    if not(st[-1] in ['1','2','3','4']):
        return 0
        
    if open_old_pronounce==1:
        if wd in rus:
            return 2
        if wd in ['嗟','瘸','靴','爹']:
            return 1
        if st[:-1] in ss:
            return 2
        
        
        if (st[-1]=='2' and st[0] in ['b','d','g','j','z']):
            return 2
        if 'ue' in st:
            return 2
            
    if st[-1] in ['1','2']:
        return 1
    
    return 2
    
# inner rhy, must obey
def checkrhyself(sentence):
    if len(sentence)==0:
        return 0
    st=sentence
    fullst=False
    while (len(st)>0 and st[-1] in [',','。','，','?','？','!','！']):
        st=st[:-1]
        fullst=True
    
    l1=pinyin(st,style=TONE3)
    if len(l1)<len(st):
        # print(l1,sentence)
        return 1
    for i in l1:
        if len(i[0])<2:
            return 1
    if len(st)<=3:
        return 2
        
      
    
    pz1=checkpz(l1[1][0],sentence[1])
    
    if len(st)>=4:
        # if len(l1[3])<1:
        #     print(sentence,l1)
        pz2=checkpz(l1[3][0],sentence[3])
        if pz2+pz1!=3:
            return 1
    if len(st)>=6:
        # if len(l1[5])<1:
        #     print(sentence,l1)
        pz3=checkpz(l1[5][0],sentence[5])
        if pz2+pz3!=3:
            return 1
    if fullst:
        if len(sentence)<6:
            return 1
        pz11=checkpz(l1[-3][0],st[-3])
        pz12=checkpz(l1[-2][0],st[-2])
        pz13=checkpz(l1[-1][0],st[-1])
        if (pz11==pz12) and (pz12==pz13):
            return 1
        
    return 2
        
            
    
    
        
        
    
def checkrhy(sentence,last,imp,req=0):
   
    while (len(sentence)>0 and (sentence[-1] in [',','。','，','?','？','!','！'])):
        sentence=sentence[:-1]
    if len(sentence)==0:
        return 0
        
    while last[-1] in [',','。','，','?','？','!','！']:
        last=last[:-1]
    l1=pinyin(sentence,style=TONE3)
    l2=pinyin(last,style=TONE3)
        #print(l1,l2)
    disobey=0
    if len(l1)!=len(sentence):
        return -1000
    for i in range(len(sentence)):
        if (i<len(l1)) and (i<len(l2)):
            st1=checkpz(l1[i][0],sentence[i])
            
            sr1=checkpz(l2[i][0],last[i])
            if (req==1 and i%2==1):
                st1=3-st1
                
            if st1+sr1!=3:
                if req==0:
                    disobey+=0.35
                if i%2==1:
                    disobey+=0.35
                    if req==1:
                        disobey+=0.2
                if i==len(l2)-1:
                    disobey+=0.65
                    if req==1:
                        disobey+=0.35
                
    disobey*=imp
    disobey=-5*disobey/len(l2)
    for i in range(len(l1)):
        for j in range(i+2,len(l1)):
            if l1[i][0][:-1]==l1[j][0][:-1]:
                disobey-=7/len(l1)
    return disobey
def checksentence(sentence,original_context,min_length,max_length,endnote,curvote=0,yayun=None):
    
    if "<|end" in sentence:
        return 1
   
    if "的" in sentence:
        return 1
    if len(sentence)==0:
        return 1
    if ((len(sentence)>max_length and not(sentence[-1] in endnote)) or len(sentence)==0) or len(sentence)>max_length+1:
        return 1
    if (sentence[-1] in endnote)and ((len(sentence)<=min_length) or (len(sentence)==7)):
        return 1
            
    if (sentence[-1] in endnote)and (sentence[:-1] in original_context):
        return 1
    
    mdisobey=0
    illegal_notes=[' ',':','《','》','‘','“','-','——','⁇','[','【','】',']','.','、','(','（',')','）','·']
    if '。' in endnote:
        illegal_notes.extend([',','，'])
    else:
        illegal_notes.append('。')
    for i in range(10):
        illegal_notes.append(str(i))
    for i in range(64,123):
        illegal_notes.append(chr(i))
    for note in illegal_notes:
        if note in sentence:
            return 1
    last=getlastsentence(original_context)
    if min_length==max_length:
        imp=1
        if (',' in last) or('，' in last):
            imp=1.5
            
        if curvote==0:
            rt=checkrhy(sentence,last,imp,req=1)
        else:
            rt=checkrhy(sentence,last,imp)
        if rt<-0.75:
            return 1
        
        
    for i in range(len(sentence)):
       # if sentence[i]=="柯":
        #    print(sentence[i],last[i],sentence[i]==last[i])
        if min_length==max_length:
            if (i<len(last)-1) and (sentence[i]==last[i]):
                #print(sentence,last)
                return 1
                
            
        
        if i<len(sentence)-1:
            if sentence[i:i+2] in original_context:
                return 1
            if sentence[i:i+2] in sentence[:i]:
                return 1
            
    
    if checkrhyself(sentence)==1:
        return 1
    cc=curvote
    if yayun is None:
        cc=0
    if (cc==1 and len(sentence)>=max_length):
        
        final1=pinyin(sentence,style=FINALS)
        #print(sentence,len(sentence),len(final1))
        if len(final1)<max_length:
            return 1
        final1=final1[max_length-1][0]
        final2=pinyin(yayun,style=FINALS)[-1][0]
        group=[['a','ia','ua'],['ai','uai','ei','ui','uei'],['an','uan','ian'],['ie','ue','ve'],
    ['ou','iu','iou'],['ang','iang','uang'],['ao','iao'],['e','o','uo'],['en','un','uen','ong','iong','in','ing','er']]
        doc=0
        if final1==final2:
            doc=1
        for i in group:
            if (final1 in i) and (final2 in i):
                doc=1
        if doc==0:
            return 1
            
            
    if (sentence[-1] in endnote):
        return 0
        
        
    return 2
    
             

    
def generate_sentence(model,tokenizer,args,device,current_tokens,mems,endnote=[",","，","?","？"],num_candidates=1,min_length=5,max_length=7,yayun=None):
    model.eval()
    with torch.no_grad():
        #index=len(tokens[0])
        mct_tree=[]
        if mems==[]:
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
        max_tries=num_candidates*120 # 应该和length关联
        curvote=1
        if ',' in endnote:
            curvote=0
        if ',' in endnote:
            endid=43359
        else:
            endid=43361
        dpcount=0
    
        tmp=args.temperature
        
        while len(final_result)==0 and tries<max_tries:
            currentid=nextid
            tries+=1
            while currentid!=-1:
                tc=torch.log(mct_tree[currentid][4])
                tc=tc+F.relu(tc-10)*1000
                logits=mct_tree[currentid][0].view(-1)-tc*0.5
                logits=logits[:50001]
                log_probs = F.softmax(logits, dim=-1)
              
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
            #print(output_tokens_list[context_length:])
            sentence = tokenizer.DecodeIds(output_tokens_list[context_length:].tolist())
            
            #print(output_tokens_list[context_length:],context_length,sentence)
            logit=mct_tree[lastid][0]
            log_probs = F.softmax(logit, dim=-1)
            log_pbs=torch.log(log_probs)
            score=log_pbs[prev].item()
            nextid=0
            ip=checksentence(sentence,original_context,min_length,max_length,endnote,curvote=curvote,yayun=yayun)
            for j in final_result:
                if j[0]==sentence:
                    ip=1
                if ('<|end' in sentence) and ('<|end' in j[0]):
                    ip=1
                    
            score=mct_tree[lastid][5]+score
            if (ip==1):
                nextid=lastid
                dpcount+=1
                max_tries+=1
                if (dpcount>=50) or (dpcount>=8 and len(sentence)<max_length):
                    nextid=0
                    dpcount=0
                mct_tree[lastid][4][prev]=100000
                continue
            dpcount=0
            if (ip==0):
                mct_tree[lastid][4][prev]=100000
                yay=yayun
                if curvote==1:
                    yay=sentence[-2]
                    
                final_result.append([copy.deepcopy(sentence),copy.deepcopy(score),copy.deepcopy(tokens),copy.deepcopy(mct_tree[lastid][1]),yay])
                #print(sentence,score)
                continue
        
           
            
                #calculate
            mct_tree[lastid][3][prev]=len(mct_tree)
            tmp=args.temperature
            if (len(sentence)>=4 or (len(sentence)==3 and max_length==5)):
                tmp=tmp*0.6
            rts=mct_tree[lastid][1]
            index=len(tokens[0])
            
            
            logits,*rts=model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                        tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                                dtype=torch.float), *rts)
            # print(len(final_result), len(mct_tree), currentid, mct_tree[lastid][5], len(sentence), len(logits), max_length, sentence)
            # print("generate sentence:{}".format(torch.cuda.memory_allocated(0)))
            logits=logits[0,-1]/tmp
            if len(sentence)==max_length:
                logits[endid]+=10
            mct_tree.append([logits,rts,tokens,-np.ones(len(logits)),torch.ones(len(logits)).cuda(),score])
            nextid=len(mct_tree)-1
        del mct_tree
        torch.cuda.empty_cache()
        #print(tries,len(final_result))
        return final_result
def getlength(str):
    w=str.replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace(' ',',').replace('！',',').replace('!',',').replace(':',',').replace(' ','')
    sp=w.split(',')
    
    return len(sp[-2])

def getlastsentence(str):
    w=str.replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace(' ',',').replace('！',',').replace('!',',').replace(':',',').replace(' ','')
    sp=w.split(',')
    fom=sp[-1]
    if len(fom)==0:
        fom=sp[-2]
    return fom+str[-1]

def get2sentencebefore(str):
    w=str.replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace(' ',',').replace('！',',').replace('!',',').replace(':',',').replace(' ','')
    sp=w.split(',')
    idk=-1
    while len(sp[idk])==0:
        idk-=1
    idk-=1
    while len(sp[idk])==0:
        idk-=1
    return sp[idk]

def check2compare(sentence1,sentence2,imp):
    s1=sentence1.replace('。','').replace('，','').replace('？','').replace('?','').replace('  ','').replace('！','').replace('!','').replace(',','')
    s2=sentence2.replace('。','').replace('，','').replace('？','').replace('?','').replace(' ','').replace('！','').replace('!','').replace(',','')
    if len(s1)!=len(s2):
        return -1000
    num=0
    for i in range(len(s1)):
        if s1[i]==s2[i]:
           num+=1
        
    score=0.5-num*num*2.5
           
    w1=pinyin(s1,style=FINALS)[-1][0]
    w2=pinyin(s2,style=FINALS)[-1][0]
    w3=pinyin(s1)[-1]
    w4=pinyin(s2)[-1]
    if (w1!=w2) or (s1[-1]==s2[-1]):
        score-=imp*0.6
    group=[['a','ia','ua'],['ai','uai','ei','ui','uei'],['an','uan','ian','ie','ue','ve'],
    ['ou','iu','iou'],['ang','iang','uang'],['ao','iao'],['e','o','uo'],['en','un','uen','ong','iong','in','ing','er']]
    if (w1!=w2)and(s1[-1]!=s2[-1]):
        for i in group:
            if (w1 in i) and (w2 in i):
                score+=imp*1
    if (w1==w2) and (w3!=w4):
        score+=imp*1
        
    return score

def check2com(sentence,org_context,imp):
    
    before2=get2sentencebefore(org_context)
    before1=getlastsentence(org_context)[:-1]
    s1=check2compare(sentence,before2,imp)
    if imp==1:
        s2=checkrhy(sentence,before1,imp+0.5,req=1)
    else:
        s2=checkrhy(sentence,before1,imp)
    sc=s1+s2+imp
    for i in range(len(sentence)-1):
        if sentence[i] in org_context:
            sc-=3
            if sentence[i:i+2] in org_context:
                sc-=5
                if (',' in sentence[i:i+2]) or ('，' in sentence[i:i+2]) or ('。' in sentence[i:i+2]) or ('？' in sentence[i:i+2]) or('?' in sentence[i:i+2]) or ('！' in sentence[i:i+2]) or ('!' in sentence[i:i+2]):
                    sc-=35
        
    return sc
    
    
    
    
    
def generate_string(model, tokenizer, args, device,title,author,desc=None,length=None,st=None):
    input_str=title+" 作者:"+author+" 体裁:诗歌 题名:"+title+" 正文: "
    if desc is not None:
        input_str=title+" 作者:"+author+" 体裁:诗歌 描述:"+desc+" 题名:"+title+" 正文: "
    #aus=author.split(' ')[1]
    input_len=len(input_str)
    context_count=0
    model.eval()
    # print("start generate string:{}".format(torch.cuda.memory_allocated(0)))
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
        eo_tokens=tokenizer.EncodeAsIds('<|endoftext|>').tokenization
        context_length = len(context_tokens)
        if context_length>=args.seq_length:
            raise InputTooLongException("the text you entered is too long, please reduce the number of characters")

        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        eo_token_tensor=torch.cuda.LongTensor(eo_tokens)
        context_length_tensor = torch.cuda.LongTensor([context_length])
        context_length = context_length_tensor[0].item()
        #tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)

        start_time = time.time()

        counter, mems = 0, []
        org_context_length = context_length
        beam_size=1
        beam_candidate=1
        beam_max=1
        max_headings=4
        final_storage=[]
        final_storage_score=[]
        step=9
        if st is None:
            st=8
        overall_score=[]
        past_beam_id=[]
        
        #print(counter,beam_tokens,beam_score)
        if length is not None:
            beam_sentences=generate_sentence(model,tokenizer,args,device,context_tokens_tensor,[],min_length=length,max_length=length,num_candidates=beam_size)
        else:
            beam_sentences=generate_sentence(model,tokenizer,args,device,context_tokens_tensor,[],num_candidates=beam_size)

        if len(beam_sentences)==0:
            raise CanNotReturnException("unable to give a reasonable result, please change the method of inquiry or contact relevant personnel for feedback")
            
        for i in range(step):
            beam_new_sentences=[]
            
            endnote=[',','，','?','？']
            if i%2==0:
                endnote=['。','?','？','！','!']
            overall_score=[]
            past_beam_id=[]
            id=0
            current_sentence=input_str+beam_sentences[0][0]
                
               #print(beam_sentences[id][0],beam_sentences[id][1])
            ini_score=beam_sentences[id][1]
            token_tensor=beam_sentences[id][2]
            mems=beam_sentences[id][3]
            
            len_sentence=getlength(beam_sentences[id][0])
                
                #print(token_tensor)
                
            gen=generate_sentence(model,tokenizer,args,device,token_tensor,mems,num_candidates=beam_candidate,endnote=endnote,min_length=len_sentence,max_length=len_sentence,yayun=beam_sentences[id][-1])
            if len(gen)==0:
                raise CanNotReturnException("unable to give a reasonable result, please change the method of inquiry or contact relevant personnel for feedback")
            jj=gen[0]
            if ('<|end' in jj[0] or i==7) :
                if (i%2==1 and i>-3):
                    del beam_sentences
                    del beam_new_sentences
                    torch.cuda.empty_cache()
                    # print(current_sentence, )
                    # print("{}, generate string:{}".format(step, torch.cuda.memory_allocated(0)))
                    return(current_sentence)
                else:
                    # print("{}, else generate string:{}".format(step, torch.cuda.memory_allocated(0)))
                    gen=generate_sentence(model,tokenizer,args,device,token_tensor,mems,num_candidates=beam_candidate,endnote=endnote,min_length=len_sentence,max_length=len_sentence,yayun=beam_sentences[id][-1])
        
            if len(gen)==0:
                raise CanNotReturnException("unable to give a reasonable result, please change the method of inquiry or contact relevant personnel for feedback")
            st=jj[0]
            # experiment shows that this is better universal,
                    
            jj[0]=beam_sentences[id][0]+jj[0]
            jj[1]=0
            #print(i,beam_sentences[id][0],jj[1])
            #print(i,jj[0],jj[1]/(i+2))
            beam_new_sentences.append(jj)
            del beam_sentences
            torch.cuda.empty_cache()
            beam_sentences=beam_new_sentences
            
            #print(i,beam_sentences[gy[-k]][0],beam_sentences[gy[-k]][1]/(i+2))
                        
            #parallel ends
            
        
        del beam_sentences
        del beam_new_sentences
        torch.cuda.empty_cache()
        # print("end generate string:{}".format(torch.cuda.memory_allocated(0)))
        raise CanNotReturnException("unable to give a reasonable result, please change the method of inquiry or contact relevant personnel for feedback")        
            

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

def set_args():
    args=get_args()
    print(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    #set up
    #print(args)
    args.deepspeed=True
    args.num_nodes=1
    args.num_gpus=1
    args.model_parallel_size=1
    args.deepspeed_config="script_dir/ds_config.json"
    args.num_layers=32
    args.hidden_size=2560
    args.load="/workspace/xuzou/useful_models/ts_human"
    #args.load="/mnt3/xinhuapoem/xinhua"
    args.num_attention_heads=32
    args.max_position_embeddings=1024
    args.tokenizer_type="ChineseSPTokenizer"
    args.cache_dir="cache"
    args.fp16=True
    args.out_seq_length=180
    args.seq_length=200
    args.mem_length=256
    args.transformer_xl=True
    args.temperature=1.2
    args.top_k=0
    args.top_p=0
    
    return args
def prepare_model():
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
    

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    args.seed=random.randint(0,1000000)
    set_random_seed(args.seed)

    #get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)
    #args.load="../ckp/txl-2.8b11-20-15-10"
    #model2=setup_model(args)
    #setting default batch size to 1
    args.batch_size = 1

    #generate samples
    return model,tokenizer,args

def generate_strs(tups):
    model,tokenizer,args=prepare_model()
    output=[]
    for tup in tups:
        #str=generate_token_tensor(str,tokenizer)
        
        output_string,output_scores=generate_string(model,tokenizer, args, torch.cuda.current_device(),tup[0],tup[1],desc=tup[2])
        list_poems=0
        
        ranklist=np.argsort(output_scores)
        best_score=output_scores[ranklist[0]]
        text_dir="poems_save/"
        already=[]
        with jsonlines.open(text_dir+tup[0]+tup[1]+'.jsonl', mode='w') as writer:
            for i in range(len(ranklist)):
                j=ranklist[i]
                if output_scores[j]<best_score+2:
                    if not(output_string[j][0:15] in already):
                        otc={}
                        otc['author']=tup[1]
                        otc['title']=tup[0]
                        otc['context']=output_string[j]
                        #print(otc)
                        writer.write(otc)
                        already.append(output_string[j][0:15])
                        
        
        
    return 0

    

def generate():
    fi=[]
    
    title_list=[["咏特朗普","咏美国前总统，民粹主义政客唐纳德 特朗普"]]
    author_list=["李白","杜甫"]
    
    for j in author_list:
        for i in title_list:
            fi.append([i[0],j,i[1]])
           
            #fi.append([i[0],j,i[1]])
    output=generate_strs(fi)
    return 0
    

def random_generate(mode=0):
    text_dir="poems_save_modernfast/"
    if mode==1:
        text_dir="poems_save_poemnetf/"
    if mode==2:
        text_dir="poems_save_hua/"
    dist=os.listdir()
    if not(text_dir[:-1] in dist):
        os.mkdir(text_dir[:-1])
    if mode==0:
        qts=open("selected_102.txt",'r')
        
        wt=qts.readlines()
        lt=[]
        for i in wt:
            
            sp=i.split()
            #print(sp)
            author="李白"
            title=sp[0]
            num_wd=int(sp[2])
            num_st=int(sp[3])
            lt.append([author,title,num_wd,num_st])
        qts.close()
    if mode==1:
        qts=open("selected_modern.txt",'r')
        
        wt=qts.readlines()
        lt=[]
        for i in wt:
            
            sp=i.split()
            if len(sp)>0:
            #print(sp)
                author="苏轼"
                title=sp[0]
                lt.append([author,title])
        qts.close()
    if mode==2:
        qts=open("mars.txt",'r')
        qts2=open("carbon.txt",'r')
        wt=qts.readlines()
        wt2=qts2.readlines()
        
        lt=[wt,wt2]
        import json
        for j in lt:
            for jj in j:
            
                if '{' in jj:
                    #print(jj)
                    sp=json.loads(jj)
                    
                    if len(sp)>0:
                    #print(sp)
                        author="李白"
                        title=sp['headLine']
                        desc=sp['content'].replace('\u3000','')
                        desc=desc.split('\n')[0]
                        lt.append([author,title,desc])
        qts.close()
        qts2.close()
        
    model,tokenizer,args=prepare_model()
    while True:
        id=random.randint(0,len(lt)-1)
        if mode==0:
            author,title,num_wd,num_st=lt[id]
        if mode==1:
            author,title=lt[id]
        if mode==2:
            author,title,desc=lt[id]
        lists=os.listdir(text_dir)
        lts=title+author+'.jsonl'
        if (lts in lists):
            continue
        #str=generate_token_tensor(str,tokenizer)
        if mode==0:
            output_string=generate_string(model, tokenizer, args, torch.cuda.current_device(),title,author,st=num_st,length=num_wd)
        if mode==1:
            output_string=generate_string(model, tokenizer, args, torch.cuda.current_device(),title,author)
        if mode==2:
            output_string=generate_string(model, tokenizer, args, torch.cuda.current_device(),title,author,desc=desc)
        output_string=[output_string]
        new_output_string=[]
        new_output_score=[]
        for i in range(len(output_string)):
            st=output_string[i].replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace('!',',').replace('！',',')
            st=st.split(',')
                #print(st,num_st)
            if mode==0:
                if len(st)-1==num_st:
                    new_output_string.append(output_string[i])
            else:
                new_output_string.append(output_string[i])
                
        if len(new_output_string)==0:
            del output_string
            continue
        
        list_poems=0
        
        ranklist=[0]
        
        already=[]
        
        with jsonlines.open(text_dir+title+author+'.jsonl', mode='w') as writer:
            for i in range(len(ranklist)):
                j=ranklist[i]
                    
                    
                otc={}
                otc['author']=author
                otc['title']=title
                otc['context']=new_output_string[j]
                        #print(otc)
                writer.write(otc)
                already.append(new_output_string[j][0:5])
        
        del output_string
        
    return 0

def poem_interface(model,tokenizer, args, device,title,author,desc=None):
    if not(desc is None):
        output_string=generate_string(model,tokenizer, args, device,title,author,desc=desc)
    else:
        output_string=generate_string(model,tokenizer, args, device,title,author)
        
    return output_string
    

def fast_poem(content,model, tokenizer, args):
    params = content.split("&&")
    title = params[0]
    author = params[1]
    desc = params[2]
    output_string = generate_string(model, tokenizer, args, torch.cuda.current_device(), title,author, desc=desc)
    # print("end", output_string)
    # print("end string:{}".format(torch.cuda.memory_allocated(0)))
    # new_output_string = []
    # new_output_score = []
    # for i in range(len(output_string)):
    #     st = output_string[i].replace('。', ',').replace('，', ',').replace('？', ',').replace('?', ',').replace('!',',').replace('！', ',')
    #     st = st.split(',')
    #     # print(st,num_st)
    #     # if len(st)-1==num_st:
    #     new_output_string.append(output_string[i])
    #     new_output_score.append(output_scores[i])

    # ranklist = np.argsort(new_output_score)
    # # print(ranklist)

    # best_pos = ranklist[0]
    # # print("output over : " + title + " " + author)
    # # print(new_output_string[best_pos])
    # return new_output_string[best_pos]
    return output_string

