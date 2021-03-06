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
from com_utils.http_utils import CanNotReturnException, InputTooLongException, IllegalParamException
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

def generate_score(model, tokenizer, args, device,  mid_str, eval_str, raw_mems=None,pen=1):
    penalty=0
    if pen==1:
        sp=mid_str.split("???")
        if len(sp)<2:
            print(sp)
        if len(sp)==0:
            return -1000
        title=sp[0]
        ee=sp[1]+eval_str
        for i in ee:
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
        del log_probs
        del log_num
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
    w=s.replace('???',',').replace('???',',').replace('???',',').replace('?',',').replace('<','').replace(' ','').replace('>','').replace('???','').replace('???','').replace('\"','').replace('???','').replace('???','').replace("???",'').replace('???','').replace('???','').replace('-','').replace('.','').replace('!',',').replace('???',',')
    return len(w)

def generate_token_tensor(str,tokenizer):
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(str).tokenization
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        return context_tokens_tensor

rus=set(['???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???','???'])
ss=set(['de','te','le','ze','ce','se','fa','fo','dei','zei','gei','hei','sei','bie','pie','mie','die','tie','nie','lie','kuo','zhuo','chuo','shuo','ruo'])
def checkpz(st,wd):
    #???????????????
    
    #?????????????????????
    if not(st[-1] in ['1','2','3','4']):
        return 0
        
    if open_old_pronounce==1:
        if wd in rus:
            return 2
        if wd in ['???','???','???','???']:
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
def checkrhyself(sentence,pro,yayun=None):
    if len(sentence)==0:
        return 0
    st=sentence
    fullst=False
    while (len(st)>0 and st[-1] in [',','???','???']):
        st=st[:-1]
        fullst=True
    
    l1=pinyin(st,style=TONE3)
    if len(l1)!=len(st):
        return 1
    for i in range(len(st)):
        pz=checkpz(l1[i][0],st[i])
        #print(i,st[i],pz)
        if not(l1[i][0][-1] in ['1','2','3','4']) and (pro[i]=='???'):
            return 1
            
        if (pz!=2) and (pro[i]=='???'):
            return 1
        if (pz!=1) and (pro[i]=='???'):
            return 1
    group=[['a','ia','ua'],['ai','uai','ei','ui','uei'],['an','uan','ian','ie','ue','ve'],
    ['ou','iu','iou'],['ang','iang','uang'],['ao','iao'],['e','o','uo'],['en','un','uen','ong','iong','in','ing','er']]
    if yayun is not None:
        if (pro[-1]=='*' and len(st)>=len(pro)-1):
        
            final1=pinyin(st,style=FINALS)[len(pro)-2][0]
            final2=pinyin(yayun,style=FINALS)[-1][0]
            doc=0
            if final1==final2:
                doc=1
            for i in group:
                if (final1 in i) and (final2 in i):
                    doc=1
            if doc==0:
                return 1
    
    
        
    return 2
    
def checksentence(sentence,original_context,min_length,max_length,endnote,pron=None,yayun=None):
    if ",," in sentence:
        return 1
    if "??????" in sentence:
        return 1
    if "<|" in sentence:
        return 1
   
    if len(sentence)==0:
        return 1
    if ((len(sentence)>max_length and not(sentence[-1] in endnote)) or len(sentence)==0) or len(sentence)>max_length+1:
        return 1
    if (sentence[-1] in endnote)and (len(sentence)<=min_length):
        return 1
            
    if (sentence[-1] in endnote)and (sentence[:-1] in original_context):
        return 1
    
    mdisobey=0
    illegal_notes=[' ',':','???','???','???','???','-','??????','???','[','???','???',']','.','???','(','???',')','???','??']
    if '???' in endnote:
        illegal_notes.extend([',','???'])
    else:
        illegal_notes.append('???')
    for i in range(10):
        illegal_notes.append(str(i))
    for i in range(64,123):
        illegal_notes.append(chr(i))
    for note in illegal_notes:
        if note in sentence:
            return 1
        
        
    for i in range(len(sentence)):
        if i<len(sentence)-3:
            if sentence[i:i+3] in original_context:
                return 1
            if sentence[i:i+2] in sentence[:i]:
                return 1
                
    if checkrhyself(sentence,pron,yayun=yayun)==1:
        return 1
    if (pron[-1]=='&' and len(sentence)>=len(pron)-1):
        final1=pinyin(sentence,style=FINALS)[len(pron)-2][0]
        final2=pinyin(original_context[-2],style=FINALS)[-1][0]
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
    

   
def generate_sentence(model,tokenizer,args,device,current_tokens,mems,endnote=[",","???","?","???"],num_candidates=10,min_length=5,max_length=7,heading=None,pron=None,yayun=None):
    #required repetition
    # add_threshold=75
    model.eval()
    if pron is not None:
        l=len(pron)
        if pron[-1] in ['*','&','^']:
            l=l-1
    min_length=l
    max_length=l
    #print(min_length,max_length,endnote,pron,yayun)
    with torch.no_grad():
        #index=len(tokens[0])
        mct_tree=[]
        
            
            
            
        if len(mems)==0:
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
        org=original_context.split('??????:')
        desc_num=len(org[0])
        add_threshold=50+desc_num
            
        if pron is not None:
            if pron[-1]=='R':
                org=original_context[:-1]
                w=len(org)-1
                while not(org[w] in [',','???','???',' ',':','???']):
                    w-=1
                context_tobe_repeated=org[w+1:]
                all_context=original_context+context_tobe_repeated+endnote[-1]
                encode_context=tokenizer.EncodeAsIds(all_context).tokenization
                context_tokens_tensor = torch.cuda.LongTensor(encode_context)
                return [[context_tobe_repeated+endnote[-1],0,context_tokens_tensor,[],yayun]]
        tokens2=[]
        rts2=[]
        logits2=[]
        add_sec=False
        if len(original_context)>add_threshold:
            add_sec=True
        if add_sec:
            org=original_context.split('??????:')
            in_st=org[0]+'??????:'+getlastsentence(original_context,length=20)
            in_st_tok=tokenizer.EncodeAsIds(in_st).tokenization
            in_tensor=torch.cuda.LongTensor(in_st_tok)
            mems2=[]
            tokens2, attention_mask2, position_ids2 = get_batch(in_tensor, device, args)
            logits2,*rts2 = model(tokens2, position_ids2, attention_mask2, *mems2)
                
                
        context_length=len(tokens[0])
        logits=logits[0,-1]
        if add_sec:
            logits=0.4*(logits+1.5*logits2[0,-1])
            
        if heading is not None:
            head=tokenizer.EncodeAsIds(heading).tokenization
            headid=head[-1]
            logits[headid]+=10000
            # print(heading,head,headid)
        #mct_structure=-np.ones(len(logits))
        #endid=tokenizer.EncodeAsIds(endnote[-1]).tokenization
       #print(endid,tokenizer.DecodeIds([endid[-1]]))
        if ',' in endnote:
            endid=43359
        else:
            endid=43361
        
        
        mct_tree.append([logits,rts,tokens,-np.ones(len(logits)),torch.ones(len(logits)).cuda(),0,tokens2,rts2,torch.zeros(len(logits)).cuda()])
        
        #print(logits.shape)
        final_result=[]
        nextid=0
        tries=0
        max_tries=num_candidates*35
        curvote=1
        if ',' in endnote:
            curvote=0
        #chainids=[]
        
        tmp=args.temperature
        dpcount=0
        while ((len(final_result)<num_candidates)and(tries<max_tries)) and (tries<4500):
            currentid=nextid
            tries+=1
            if len(final_result)==0:
                max_tries+=1
        
            while currentid!=-1:
                tc=torch.log(mct_tree[currentid][4])
                tc=tc+F.relu(tc-10)*1000
                logits=mct_tree[currentid][0].view(-1)-tc*0.75
                logits=logits[:50001]
                
                log_probs = F.softmax(logits, dim=-1)
                
                pr=torch.multinomial(log_probs,num_samples=1)[0]
                #pr=torch.argmax(logits)
                prev=pr.item()
                
                #print(logits.shape,currentid,prev)
                mct_tree[currentid][4][prev]+=1
                lastid=currentid
                #chainids.append([currentid,prev])
                currentid=int(mct_tree[currentid][3][prev])
            #start from lastid & currentid
            
            cqs=mct_tree[lastid][2]
            #print(pr)
            tokens = torch.cat((cqs, pr.unsqueeze(0).view(1, 1)), dim=1)
            ll=mct_tree[lastid][8]+0.0
            ll[prev]+=1
            output_tokens_list = tokens.view(-1).contiguous()
            if add_sec:
                cqs=mct_tree[lastid][6]
                tokens2=torch.cat((cqs, pr.unsqueeze(0).view(1, 1)), dim=1)
                
            #if max_length==min_length:
             #   print(min_length,output_tokens_list,context_length)
            #print(output_tokens_list[context_length:])
            sentence = tokenizer.DecodeIds(output_tokens_list[context_length:].tolist())
        
            logit=mct_tree[lastid][0]
            log_probs = F.softmax(logit, dim=-1)
            log_pbs=torch.log(log_probs)
            score=log_pbs[prev].item()
            nextid=0
            
            ip=checksentence(sentence,original_context,min_length,max_length,endnote,pron=pron,yayun=yayun)
            for j in final_result:
                if j[0]==sentence:
                    ip=1
                if ('<|end' in sentence) and ('<|end' in j[0]):
                    ip=1
                    
            score=mct_tree[lastid][5]+score
            #if tries in [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]:
            if (tries>3999 and tries<4100):
                if tries%100==0:
                    pass
                    # print(original_context)
                # print(tries,sentence,score,min_length,max_length,pron,yayun,ip)
                
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
                if ('*' in pron) or ('^' in pron):
                    yay=sentence[-2]
                else:
                    yay=yayun
               
                final_result.append([copy.deepcopy(sentence),copy.deepcopy(score),copy.deepcopy(tokens),copy.deepcopy(mct_tree[lastid][1]),yay])
                #print(sentence,score)
                continue
            
           
            
                #calculate
            mct_tree[lastid][3][prev]=len(mct_tree)
            tmp=args.temperature
            if (len(sentence)>=max_length*0.6):
                tmp=tmp*0.6
            rts=mct_tree[lastid][1]
            index=len(tokens[0])
            
            
            logits,*rts=model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                        tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                                dtype=torch.float), *rts)
            if add_sec:
                index2=len(tokens2[0])
                rts2=mct_tree[lastid][7]
                logits2,*rts2=model(tokens2[:, index2 - 1: index2], tokens2.new_ones((1, 1)) * (index2 - 1),
                        tokens2.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                                dtype=torch.float), *rts2)
                
                
            logits=logits[0,-1]/tmp
            if add_sec:
                logits=0.4*(logits+1.5*logits2[0,-1]/tmp)
                
            if len(sentence)==max_length:
                logits[endid]+=15
            logits=logits-ll
            
            mct_tree.append([logits,rts,tokens,-np.ones(len(logits)),torch.ones(len(logits)).cuda(),score,tokens2,rts2,ll])
            nextid=len(mct_tree)-1
       # print(tries,len(final_result))
        del mct_tree
        del logits
        del rts
        del logits2
        del rts2
        torch.cuda.empty_cache()
        #print(tries,len(final_result))
        return final_result
def getlength(str):
    w=str.replace('???',',').replace('???',',').replace('???',',').replace('?',',').replace(' ',',').replace('???',',').replace('!',',').replace(':',',').replace(' ','')
    sp=w.split(',')
    
    return len(sp[-2])

def getlastsentence(str,length=2):
    loc=len(str)-length
    while ((loc>0) and not(str[loc]=='???')):
        loc-=1
    
    return str[loc:]

def get2sentencebefore(str):
    w=str.replace('???',',').replace('???',',').replace('???',',').replace('?',',').replace(' ',',').replace('???',',').replace('!',',').replace(':',',').replace(' ','')
    sp=w.split(',')
    idk=-1
    while len(sp[idk])==0:
        idk-=1
    idk-=1
    while len(sp[idk])==0:
        idk-=1
    return sp[idk]

def check2compare(sentence1,sentence2,imp):
    s1=sentence1.replace('???','').replace('???','').replace('???','').replace('?','').replace('  ','').replace('???','').replace('!','').replace(',','')
    s2=sentence2.replace('???','').replace('???','').replace('???','').replace('?','').replace(' ','').replace('???','').replace('!','').replace(',','')
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
    sc=0
    for i in range(len(sentence)-1):
        if sentence[i] in org_context:
            sc-=3
            if sentence[i:i+2] in org_context:
                sc-=5
                if (',' in sentence[i:i+2]) or ('???' in sentence[i:i+2]) or ('???' in sentence[i:i+2]) or ('???' in sentence[i:i+2]) or('?' in sentence[i:i+2]) or ('???' in sentence[i:i+2]) or ('!' in sentence[i:i+2]):
                    sc-=35
        
    return sc
    
    
def get_pron():
    file='cipai.txt'
    f=open(file,'r')
    lines=f.readlines()
    cp={}
    alllist=[]
    for line in lines:
        linsp=line.split(':')
        if len(linsp)>1:
        #shuangdiao
            cp[linsp[0]]=linsp[1].replace('\n','')
            alllist.append(linsp[0])
    return cp,alllist
    
def generate_string(model, tokenizer, args, cploader, device,title,author,cipai,desc=None,length=None,headings=None):
    input_str=title+' ??????:'+cipai+" ??????:"+author+"  ??????:"+title+" ??????:"+desc+" ??????: "

    hd=0
    if headings is not None:
        hd=1
        input_str=input_str+headings
    #aus=author.split(' ')[1]
    if cipai in cploader:
        pron=cploader[cipai]
    else:
        raise IllegalParamException("invalid cipai")
    allowdouble=0
    if pron[-1]=='D':
        allowdouble=1
        pron=pron[:-1]
    pron_txt=pron.replace('???',',').replace('???',',').split(',')
    while len(pron_txt[-1])==0:
        pron_txt=pron_txt[:-1]
    pron_chr=[]
    for i in range(len(pron)):
        if pron[i] in [',','???','???']:
            pron_chr.append(pron[i])
    yayun=None
    if hd==1:
        if '*' in pron_txt[0]:
            yayun=headings[-1]
        input_str=input_str+pron_chr[0]


    # print(pron_txt,pron_chr)
    input_len=len(input_str)
    context_count=0
    model.eval()
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
        beam_size=10
        beam_candidate=7
        beam_max=2
        max_headings=5
        final_storage=[]
        final_storage_score=[]
        step=len(pron_txt)+1-hd
        if allowdouble==1:
            step=2*len(pron_txt)-hd

        overall_score=[]
        past_beam_id=[]
        #print(counter,beam_tokens,beam_score)
        # hd=None
        # if headings is not None:
        #     if len(headings)>1:
        #         hd=headings[0]
        pron_set=[pron_chr[hd]]
        if '???' in pron_set:
            pron_set.append(',')
        
        beam_sentences=generate_sentence(model,tokenizer,args,device,context_tokens_tensor,[],num_candidates=beam_size*5,endnote=pron_set,pron=pron_txt[hd],yayun=yayun)

        for w in range(len(beam_sentences)):
            if '<|end' in beam_sentences[w][0]:
                continue
            if hd==0:
                input='???'+beam_sentences[w][0]+'????????????????????????'+cipai+'??'
            else:
                input='???'+headings+pron_chr[0]+beam_sentences[w][0]+'????????????????????????'+cipai+'??'
            output_str=title+'???'
            score1=generate_score(model,tokenizer,args,device,input,output_str)
            '''
            input='???'+beam_sentences[w][0]+'??????????????????'
            output_str=aus
            score2=generate_score(model,tokenizer,args,device,input,output_str)
            '''
            ss=-beam_sentences[w][1]/len(beam_sentences[w][0])-8
            iscore=score1-0.45*(np.abs(ss)+ss)
            beam_sentences[w][1]=iscore
            #print(beam_sentences[w][0],beam_sentences[w][1])
            overall_score.append(iscore.cpu())
            past_beam_id.append(w)
            
        gy=np.argsort(overall_score)
        k=0
        sumbeam=np.zeros(100)
        
        gym=[]
        num=0
        while (num<beam_size)and (k<=len(gy)):
           k+=1
           if (k<len(gy) and sumbeam[past_beam_id[gy[-k]]]<beam_max):
            sumbeam[past_beam_id[gy[-k]]]+=1
            gym.append(gy[-k])
            num+=1
        best_score=-1000
        best_pos=0
        for i in range(step):
            beam_new_sentences=[]
            
            overall_score=[]
            past_beam_id=[]
            size=beam_size
            if len(gym)<size:
                size=len(gym)
            if size==0:
                del beam_sentences
                del beam_new_sentences
                torch.cuda.empty_cache()
                return final_storage,final_storage_score
            ini_score=beam_sentences[gym[0]][1]/(i+1)
            # early stopping
           
            
            for w in range(size):
                id=gym[w]
                current_sentence=input_str+beam_sentences[id][0]
                
               #print(beam_sentences[id][0],beam_sentences[id][1])
                ini_score=beam_sentences[id][1]
                token_tensor=beam_sentences[id][2]
                mems=beam_sentences[id][3]
            
                len_sentence=getlength(beam_sentences[id][0])
                
                #print(token_tensor)
                # hd=None
                # if len(headings)>=i+2:
                #     hd=headings[i+1]
                
                pron_set=[pron_chr[(i+1+hd)%len(pron_txt)]]
                if '???' in pron_set:
                    pron_set.append(',')
                gen=generate_sentence(model,tokenizer,args,device,token_tensor,mems,num_candidates=beam_candidate,endnote=pron_set,min_length=len_sentence,max_length=len_sentence,pron=pron_txt[(i+1+hd)%len(pron_txt)],yayun=beam_sentences[id][-1])
                if ((i+hd)%len(pron_txt)==len(pron_txt)-1):
                    sc=beam_sentences[id][1]/(i+1) #prioritize short poems
                    sc=sc.item()
                    if sc>best_score:
                        best_score=sc
                        best_pos=len(final_storage)-1
                    if sc>best_score-1:
                        sc=np.abs(sc)
                        final_storage.append(copy.deepcopy(current_sentence[input_len:]))
                        final_storage_score.append(sc)
                        print(current_sentence,final_storage_score[-1])
                    
            
                for jj in gen:
                    if 'end' in jj[0]:
                        continue
        
                    # experiment shows that this is better universal,
                    st=getlastsentence(beam_sentences[id][0])
                    
                    input='???'+st+jj[0]+'????????????????????????'+cipai+' '
                    output_str=title+'???'
                    
                    score1=generate_score(model,tokenizer,args,device,input,output_str)
                    '''
                    input='???'+jj[0]+'?????????????????????'
                    output_str=st+'???'
                    score2=generate_score(model,tokenizer,args,device,input,output_str,pen=0)
                    '''
                    factor=0
                    
                   
                    iscore=score1*(1-factor)
                    if i>=1:
                        imp=1
                        if i%2==0:
                            imp+=1.5
                        scorem=check2com(jj[0],beam_sentences[id][0],imp)
                        
                        iscore+=scorem
                        
                    
                    jj[0]=beam_sentences[id][0]+jj[0]
                    jj[1]=iscore+ini_score
                    #print(jj[0],jj[1])
                    beam_new_sentences.append(jj)
                    overall_score.append(jj[1].cpu())
                    past_beam_id.append(w)
            del beam_sentences
            torch.cuda.empty_cache()
            beam_sentences=beam_new_sentences
            gy=np.argsort(overall_score)
            sumbeam=np.zeros(100)
            sumheading={}
            k=0
            gym=[]
            num=0
            while (num<beam_size) and (k+1<len(past_beam_id)):
                k+=1
                
                if sumbeam[past_beam_id[gy[-k]]]<beam_max:
                    wd=beam_sentences[gy[-k]][0][:5]
                    
                    if (not(wd in sumheading)) or (sumheading[wd]<max_headings):
                        if not(wd in sumheading):
                            sumheading[wd]=1
                        else:
                            sumheading[wd]+=1
                        sumbeam[past_beam_id[gy[-k]]]+=1
                        gym.append(gy[-k])
                        num+=1
                        # i,beam_sentences[gy[-k]][0],beam_sentences[gy[-k]][1]/(i+2))
                        
                
            
        
        del beam_sentences
        del beam_new_sentences
        torch.cuda.empty_cache()
        
        return final_storage,final_storage_score
        
            

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
    #args.load="/mnt3/ckp"
    args.load="/mnt3/ckp/checkpoint_sc/new"
    args.num_attention_heads=32
    args.max_position_embeddings=1024
    args.tokenizer_type="ChineseSPTokenizer"
    args.cache_dir="cache"
    args.fp16=True
    args.out_seq_length=180
    args.seq_length=200
    args.mem_length=256
    args.transformer_xl=True
    args.temperature=1.25
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
    
    title_list=[["????????????","?????????????????????????????????????????????????????????"]]
    title_list=[["?????????","???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????"]]
    title_list=[["??????",""]]
    author_list=["??? ?????????","??? ??????","??? ??????","??? ??????"]
    
    for j in author_list:
        for i in title_list:
            
            
            
            fi.append([i[0],j,i[1]])
           
            #fi.append([i[0],j,i[1]])
    output=generate_strs(fi)
    return 0
    

def random_generate(mode=0):
    text_dir="songci/"
    if mode==1:
        text_dir="songci_save2/"
    cp,cplist=get_pron()
    dist=os.listdir()
    if not(text_dir[:-1] in dist):
        os.mkdir(text_dir[:-1])
    if mode==0:
        qts=open("selected_102.txt",'r')
        
        wt=qts.readlines()
        lt=[]
        for i in wt:
            if len(i)>0:
                sp=i.split()
                #print(sp)
                author="??????"
                title=sp[0]
                num_wd=int(sp[2])
                num_st=int(sp[3])*2
                lt.append([author,title,num_wd,num_st])
        qts.close()
    if mode==1:
        qts=open("index.txt",'r')
        
        wt=qts.readlines()
        lt=[]
        for i in wt:
            
            sp=i.split()
            if len(sp)>0:
            #print(sp)
                author="?????????"
                title=sp[0]
                cipai=random.choice(cplist)
                lt.append([author,title,cipai])
        qts.close()
        
    model,tokenizer,args=prepare_model()
    while True:
        id=random.randint(0,len(lt)-1)
        if mode==0:
            author,title,num_wd,num_st=lt[id]
        if mode==1:
            author,title,cipai=lt[id]
        lists=os.listdir(text_dir)
        lts=title+cipai+'.jsonl'
        if (lts in lists):
            continue
        #str=generate_token_tensor(str,tokenizer)
        
        if mode==1:
            output_string,output_scores=generate_string(model, tokenizer, args,cp, torch.cuda.current_device(),title,author,cipai)
       
        new_output_string=[]
        new_output_score=[]
        for i in range(len(output_string)):
            st=output_string[i].replace('???',',').replace('???',',').replace('???',',').replace('?',',').replace('!',',').replace('???',',')
            st=st.split(',')
                #print(st,num_st)
            if mode==0:
                if len(st)-1==num_st:
                    new_output_string.append(output_string[i])
                    new_output_score.append(output_scores[i])
            else:
                new_output_string.append(output_string[i])
                new_output_score.append(output_scores[i])
                
        if len(new_output_string)==0:
            del output_string
            del output_scores
            continue
        
        list_poems=0
        
        ranklist=np.argsort(new_output_score)
        best_score=new_output_score[ranklist[0]]
        
        already=[]
        
        with jsonlines.open(text_dir+title+cipai+'.jsonl', mode='w') as writer:
            for i in range(len(ranklist)):
                j=ranklist[i]
                if new_output_score[j]<best_score+2:
                    if not(new_output_string[j][0:5] in already):
                    
                        otc={}
                        otc['author']=author
                        otc['title']=title
                        otc['cipai']=cipai
                        otc['context']=new_output_string[j]
                        #print(otc)
                        writer.write(otc)
                        already.append(new_output_string[j][0:5])
        
        del output_string
        del output_scores
        
    return 0

def write_poem_song(content, model, tokenizer, args) :
    str_list = content.split("&&")
    title = str_list[0]
    author = str_list[1]
    desc = str_list[2]
    cipai = str_list[3]
    cp,cplist=get_pron()
    output_string,output_scores = generate_string(model, tokenizer, args, cp, torch.cuda.current_device(), title, author, cipai=cipai, desc=desc)
    if len(output_string)==0:
        raise CanNotReturnException("unable to give a reasonable result, please change the method of inquiry or contact relevant personnel for feedback")
    new_output_string = []
    new_output_score = []
    for i in range(len(output_string)):
        st = output_string[i].replace('???', ',').replace('???', ',').replace('???', ',').replace('?', ',').replace('!',',').replace('???', ',')
        st = st.split(',')
        new_output_string.append(output_string[i])
        new_output_score.append(output_scores[i])

    ranklist = np.argsort(new_output_score)
    best_pos = ranklist[0]
    return new_output_string[best_pos]
    
