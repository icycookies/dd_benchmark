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
            # print(iteration)
            path = os.path.join(args.load, str(iteration), "mp_rank_00_model_states.pt")
            checkpoint = torch.load(path)
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

def generate_score(model, tokenizer, args, device,  mid_str, eval_str, raw_mems=None):

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
        
    return sumlognum.cpu()
        
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
    if (sentence[-1] in endnote)and (len(sentence)<=min_length):
        return 1
            
    if (sentence[-1] in endnote)and (sentence[:-1] in original_context):
        return 1
    if (len(sentence)>4) and (sentence in original_context):
        return 1
    if (len(sentence)>7 and (sentence[-7:] in original_context)):
        return 1
    unclear_chr=['^','|']
    for chr in unclear_chr:
        if chr in sentence:
            return 1
    if (sentence[-1] in endnote):
        return 0
        
        
    return 2
    
             

    
def generate_sentence(model,tokenizer,args,device,current_tokens,mems,endnote=[",","，","。","!","！","——",":","?","？",">"," "],num_candidates=10,min_length=3,max_length=19):
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
        
        while (len(final_result)<num_candidates)and(tries<max_tries) :
            currentid=nextid
            tries+=1
            while currentid!=-1:
                tc=torch.log(mct_tree[currentid][4])
                tc=tc+F.relu(tc-10)*1000
                logits=mct_tree[currentid][0].view(-1)-tc*0.5
                logits=logits[:50001]
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
            #print(output_tokens_list[context_length:])
            sentence = tokenizer.DecodeIds(output_tokens_list[context_length:].tolist())
            
            #print(output_tokens_list[context_length:],context_length,sentence)
            logit=mct_tree[lastid][0]
            log_probs = F.softmax(logit, dim=-1)
            log_pbs=torch.log(log_probs)
            score=log_pbs[prev].item()
            nextid=0
            ip=checksentence(sentence,original_context,min_length,max_length,endnote)
            for j in final_result:
                if j[0]==sentence:
                    ip=1
                if ('<|end' in sentence) and ('<|end' in j[0]):
                    ip=1
                    
            score=mct_tree[lastid][5]+score
            if (ip==1):
                mct_tree[lastid][4][prev]=10000
                continue
            if (ip==0):
                mct_tree[lastid][4][prev]=10000
                final_result.append([copy.deepcopy(sentence),copy.deepcopy(score),copy.deepcopy(tokens),copy.deepcopy(mct_tree[lastid][1])])
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
            
            mct_tree.append([logits,rts,tokens,-np.ones(len(logits)),torch.ones(len(logits)).cuda(),score])
            nextid=len(mct_tree)-1
        del mct_tree
        torch.cuda.empty_cache()
        #print(tries,len(final_result))
        return final_result
def getlength(str):
    w=str.replace('。',',').replace('，',',').replace('？',',').replace('?',',').replace('<n>',',').replace('！',',').replace('!',',').replace(':',',').replace(' ','')
    sp=w.split(',')
    
    return len(sp[-2])

def getlastsentence(str):
    signal=['。',',','，','!','！','?','？',' ']
    signal2=['：',':']
    fom=''
    sig1=0
    sig2=0
    nowplace=1
    while True:
        nowplace+=1
        if len(str)<nowplace:
            return str
        if str[-nowplace] in signal:
            if nowplace>100:
                return str[-nowplace+1:]
        if str[-nowplace] in signal2:
            return str[-nowplace+1:]
        
    return 0

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
    s1=sentence1.replace('。','').replace('，','').replace('？','').replace('?','').replace(' ','').replace('！','').replace('!','').replace(',','')
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
    if (w1!=w2) or (s1[-1]==s2[-1]):
        score-=imp*0.6
    group=[['ei','ui'],['ou','iu'],['ie','ue'],['en','un'],['ong','eng'],['en','eng'],['in','ing'],['an','ang'],['en','eng']]
    if (w1!=w2)and(s1[-1]!=s2[-1]):
        for i in group:
            if (w1 in i) and (w2 in i):
                score+=imp*0.5
    if (w1==w2) and (s1[-1]!=s2[-1]):
        score+=imp*0.8
        
    return score
    
    
    
    
def generate_string(model, tokenizer, args, device,title,desc=''):
    # input_str="标题："+title+" 描述："+desc+" 内容："
    if desc=='':
        input_str="标题："+title+" 内容："
    else:
        input_str="标题："+title+" 内容："+desc
    #aus=author.split(' ')[1]
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
        final_storage=[]
        final_storage_score=[]
        step=80
        overall_score=[]
        past_beam_id=[]
        #print(counter,beam_tokens,beam_score)
        beam_sentences=generate_sentence(model,tokenizer,args,device,context_tokens_tensor,[],num_candidates=beam_size*3)
        for w in range(len(beam_sentences)):
            if '<|end' in beam_sentences[w][0]:
                continue
            st=beam_sentences[w][0]
            input='内容：'+st+' 标题：'
            output_str=title
            # input='"'+st+'"出自'
            # output_str='议论文《'+title+'》'
            score1=generate_score(model,tokenizer,args,device,input,output_str)

            ss=-beam_sentences[w][1]/len(beam_sentences[w][0])-7
            iscore=score1-1.5*(np.abs(ss)+ss)
            beam_sentences[w][1]=iscore
            #print(beam_sentences[w][0],beam_sentences[w][1])
            overall_score.append(iscore)
            past_beam_id.append(w)
            
        gy=np.argsort(overall_score)
        k=0
        sumbeam=np.zeros(100)
        gym=[]
        num=0
        while (num<beam_size)and (k<len(gy)):
           k+=1
           if sumbeam[past_beam_id[gy[-k]]]<beam_max:
            sumbeam[past_beam_id[gy[-k]]]+=1
            gym.append(gy[-k])
            num+=1
        best_score=-1000
        best_pos=0
        for i in range(step):
            if ((best_score>-1000) and (i>65))or(len(final_storage)>20):
                del beam_sentences
                del beam_new_sentences
                torch.cuda.empty_cache()
                return final_storage,final_storage_score
            if i>20:
                beam_size=7
                beam_candidate=5
            if i>40:
                beam_size=5
                beam_candidate=4
                
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
            
            
            if ini_score<best_score-2:
                del beam_sentences
                del beam_new_sentences
                torch.cuda.empty_cache()
                
                return final_storage,final_storage_score
            
            for w in range(size):
                id=gym[w]
                current_sentence=input_str+beam_sentences[id][0]
                
                #print(beam_sentences[id][0],beam_sentences[id][1])
                ini_score=beam_sentences[id][1]
                token_tensor=beam_sentences[id][2]
                mems=beam_sentences[id][3]
            
                
               
                #print(token_tensor)
                gen=generate_sentence(model,tokenizer,args,device,token_tensor,mems,num_candidates=beam_candidate)
                for jj in gen:
                    if (('<|end' in jj[0])) or (i>25):
                        if (not(current_sentence[-1] in [',','，','、']) and (i>8)) :
                            final_storage.append(copy.deepcopy(current_sentence[input_len:]))
                            sc=beam_sentences[id][1]/(i+1) #prioritize short poems
                            sc=sc.item()
                            length=getlength(current_sentence[input_len:])
                            # sc-=np.abs(length-500)*0.002
                            sc-=np.abs(length-200)*0.002
                            if sc>best_score:
                                best_score=sc
                                best_pos=len(final_storage)-1
                                sc=np.abs(sc)
                                final_storage_score.append(sc)
                                # print(current_sentence,final_storage_score[-1])
                            
                        continue
                    st=jj[0]
                    # experiment shows that this is better universal,
                    st=getlastsentence(beam_sentences[id][0])+jj[0]
                    # input='"'+st+'"出自'
                    
                    # output_str='议论文《'+title+'》'
                    input='内容：'+st+' 标题：'
                    
                    output_str=title
                    
                    score1=generate_score(model,tokenizer,args,device,input,output_str)
                
                   
                    
                    ss=-jj[1]/len(jj[0])-7
                    iscore=score1-1.5*(np.abs(ss)+ss)
                    
                        
                    #print(i,beam_sentences[id][0],before,jj[0])
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
                
            
        if len(final_storage)==0:
            raise CanNotReturnException("unable to give a reasonable result, please change the method of inquiry or contact relevant personnel for feedback")

        
        max_score=np.argmax(final_storage_score)
        
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
    args.load="/workspace/xuzou/useful_models/xinhua"
    args.num_attention_heads=32
    args.max_position_embeddings=1024
    args.tokenizer_type="ChineseSPTokenizer"
    args.cache_dir="cache"
    args.fp16=True
    args.out_seq_length=512
    args.seq_length=200
    args.mem_length=256
    args.transformer_xl=True
    args.temperature=1
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
        output_string,output_scores=generate_string(model,tokenizer, args, torch.cuda.current_device(),tup[0],desc=tup[1])
        list_poems=0
        
        
                        
                        
        
    return 0

    

def generate():
    fi=[]
    #title_list=["咏希拉里","咏拜登","咏蔡英文","早春","登黄鹤楼","上甘岭","赠抗疫英雄"]
    title_list=[["天问一号携祝融号成功着陆火星","5月15日，天问一号着陆巡视器成功着陆于火星乌托邦平原南部预选着陆区，我国首次火星探测任务着陆火星取得圆满成功。"]]
    title_list=[["2022年中国实现新冠病毒群体免疫",""]]
    title_list=[["2060年中国实现碳中和, 实现二氧化碳“零排放","碳中和是指企业、团体或个人测算在一定时间内直接或间接产生的温室气体排放总量，然后通过植物造树造林、节能减排等形式，抵消自身产生的二氧化碳排放量。"]]
    title_list=[["智源大会开幕致辞","HI，各位嘉宾，欢迎来到智源大会。"]]
    title_list=[["体育与国家","人的身体会天天变化。目不明可以明，耳不聪可以聪。生而强者如果滥用其强，即使是至强者，最终也许会转为至弱；而弱者如果勤自锻炼，增益其所不能，久之也会变而为强。因此，“生而强者不必自喜也，生而弱者不必自悲也。吾生而弱乎，或者天之诱我以至于强，未可知也。"]]

    output=generate_strs(title_list)
    return 0
    

def random_generate():
    text_dir="xinhua_new3/"
    
    
    dist=os.listdir()
    if not(text_dir[:-1] in dist):
        os.mkdir(text_dir[:-1])

   
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
                        
                    question=sp['headLine']
                       
                    lt.append(question)
    qts.close()
    qts2.close()
    '''
    wt=qts.readlines()
    lt=[]
    for i in wt:
        sp=i.replace('\n','')
        question=sp
        lt.append(question)
    qts.close()
    '''
    model,tokenizer,args=prepare_model()
    while True:
        id=random.randint(0,len(lt)-1)
        
        question=lt[id]
        # print(question)
        lists=os.listdir(text_dir[:-1])
        lts=question+'.jsonl'
        lts=lts.replace("/","")
        if (lts in lists):
            continue
        #str=generate_token_tensor(str,tokenizer)
        
        output_string,output_scores=generate_string(model, tokenizer, args, torch.cuda.current_device(),question)
       
       
        
        list_poems=0
        
        ranklist=np.argsort(output_scores)
        best_score=output_scores[ranklist[0]]
        
        already=[]
        
        with jsonlines.open(text_dir+lts, mode='w') as writer:
            
            j=ranklist[0]
    
                   
                    
            otc={}
            otc['question']=question
            otc['context']=output_string[j]
                        #print(otc)
            writer.write(otc)
                        
        
        del output_string
        del output_scores
        
    return 0


def news_limit_refined(model,tokenizer,args, title, desc):
    output_string,output_scores=generate_string(model, tokenizer, args, torch.cuda.current_device(), title, desc)
    if len(output_string)==0:
        raise CanNotReturnException("unable to give a reasonable result, please change the method of inquiry or contact relevant personnel for feedback")

    ranklist=np.argsort(output_scores)
    res = output_string[ranklist[0]]
    res = res.replace("<n>", "\n")
    return res
# generate()
