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
open_old_pronounce = 0
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
from pypinyin import pinyin, FINALS, FINALS_TONE, TONE3
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


def generate_score(model, tokenizer, args, device, input_str, eval_str):
    # penalty on same wor
    context_count = 0
    model.eval()
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(input_str).tokenization
        eval_tokens = tokenizer.EncodeAsIds(eval_str).tokenization
        if len(context_tokens) == 0:
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
            # logits /= args.temperature
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

        del logits
        del mems
        torch.cuda.empty_cache()
        return sumlognum


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


def checklength(s):
    w = s.replace('。', ',').replace('，', ',').replace('？', ',').replace('?', ',').replace('<', '').replace(' ',
                                                                                                           '').replace(
        '>', '').replace('《', '').replace('》', '').replace('\"', '').replace('“', '').replace('”', '').replace("‘",
                                                                                                               '').replace(
        '‘', '').replace('、', '').replace('-', '').replace('.', '').replace('!', ',').replace('！', ',')
    return len(w)


def generate_token_tensor(str, tokenizer):
    with torch.no_grad():
        context_tokens = tokenizer.EncodeAsIds(str).tokenization
        context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
        return context_tokens_tensor


rus = set(
    ['八', '搭', '塌', '邋', '插', '察', '杀', '煞', '夹', '俠', '瞎', '辖', '狹', '匣', '黠', '鸭', '押', '压', '刷', '刮', '滑', '猾', '挖',
     '蜇', '舌', '鸽', '割', '胳', '搁', '瞌', '喝', '合', '盒', '盍', '曷', '貉', '涸', '劾', '核', '钵', '剝', '泼', '摸', '脱', '托', '捋',
     '撮', '缩', '豁', '活', '切', '噎', '汁', '织', '隻', '掷', '湿', '虱', '失', '十', '什', '拾', '实', '食', '蝕', '识', '石', '劈', '霹',
     '滴', '踢', '剔', '屐', '积', '激', '击', '漆', '吸', '息', '媳', '昔', '席', '锡', '檄', '觋', '揖', '一', '壹', '扑', '匍', '仆', '弗',
     '紱', '拂', '福', '蝠', '幅', '辐', '服', '伏', '茯', '督', '突', '秃', '俗', '出', '蜀', '窟', '哭', '忽', '惚', '斛', '鹄', '屋', '屈',
     '诎', '曲', '戌', '拍', '塞', '摘', '拆', '黑', '勺', '芍', '嚼', '粥', '妯', '熟', '白', '柏', '伯', '薄', '剥', '摸', '粥', '轴', '舳',
     '妯', '熟', '角', '削', '学'])
ss = set(['de', 'te', 'le', 'ze', 'ce', 'se', 'fa', 'fo', 'dei', 'zei', 'gei', 'hei', 'sei', 'bie', 'pie', 'mie', 'die',
          'tie', 'nie', 'lie', 'kuo', 'zhuo', 'chuo', 'shuo', 'ruo'])


def checkpz(st, wd):
    # 入声字判断

    # 轻声按失败算。
    if not (st[-1] in ['1', '2', '3', '4']):
        return 2

    if open_old_pronounce == 1:
        if wd in rus:
            return 2
        if wd in ['嗟', '瘸', '靴', '爹']:
            return 1
        if st[:-1] in ss:
            return 2

        if (st[-1] == '2' and st[0] in ['b', 'd', 'g', 'j', 'z']):
            return 2
        if 'ue' in st:
            return 2

    if st[-1] in ['1', '2']:
        return 1

    return 2


def checksc(model, tokenizer, args, device, next, prop):
    if len(next) == 0:
        return 0
    org_length = len(prop)
    l1 = pinyin(prop, style=TONE3)
    l2 = pinyin(next, style=TONE3)

    if len(l1) != len(prop):
        return -10000
    if len(l2) != len(next):
        return -10000
    if len(next) > len(prop):
        return -10000
    cr1d = 0
    for i in range(len(prop)):
        for j in range(i):
            if l1[i][0][:-1] == l1[j][0][:-1]:
                cr1d += 1
    prop = prop[0:len(next)]
    tone = checkpz(l1[len(next) - 1][0], prop[len(next) - 1])
    # 仄起平收
    updown = 0
    if tone == 2:
        updown = 1

    penalty = 0

    for i in range(len(prop)):
        for j in range(i):
            if l1[i][0][:-1] == l1[j][0][:-1]:

                if l2[i][0][:-1] != l2[j][0][:-1]:
                    return -10000
            else:
                if l2[i][0][:-1] == l2[j][0][:-1]:
                    return -10000

            if prop[i] != prop[j]:
                if next[i] == next[j]:
                    return -10000

    for i in range(len(prop)):
        if len(l1[i][0]) < 2:
            return -10000
        if len(l2[i][0]) < 2:
            return -10000
        if l1[i][0][:-1] == l2[i][0][:-1]:
            return -10000
        tone1 = checkpz(l1[i][0], prop[i])
        tone2 = checkpz(l2[i][0], next[i])
        if tone1 + tone2 != 3:
            if (org_length <= 7 and cr1d == 0):
                return -10000
            else:
                penalty += 3
            if i % 2 == 1:
                if cr1d <= 2:
                    return -10000
                else:
                    penalty += 5

            if i == len(l1) - 1:
                return -10000
    # samecheck
    if len(l1) != len(l2):
        return 0

    if updown == 0:
        pro = "上联: " + next + " 下联: "
        con = prop
        pro2 = "下联: " + prop + " 上联: "
        con2 = next
    else:
        pro = "下联: " + next + " 上联: "
        con = prop
        pro2 = "上联: " + prop + " 下联: "
        con2 = next

    score1 = generate_score(model, tokenizer, args, device, pro, con)
    score2 = generate_score(model, tokenizer, args, device, pro2, con2)
    score = score1 + score2 - penalty
    return score


def generate_sentence(model, tokenizer, args, device, input,max_tries=1000):
    py1 = pinyin(input, style=TONE3)
    py = checkpz(py1[-1][0], input[-1])
    if py == 1:
        input_str = "下联: " + input + " 上联:"
    else:
        input_str = "上联: " + input + " 下联:"

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

        start_time = time.time()

        counter, mems = 0, []

        tokens, attention_mask, position_ids = get_batch(context_tokens_tensor, device, args)
        logits, *rts = model(tokens, position_ids, attention_mask, *mems)

        output_tokens_list = tokens.view(-1).contiguous()
        original_context = tokenizer.DecodeIds(output_tokens_list.tolist())
        context_length = len(tokens[0])
        logits = logits[0, -1]
        # mct_structure=-np.ones(len(logits))
        mct_tree.append([logits, rts, tokens, -np.ones(len(logits)), torch.ones(len(logits)).cuda(),
                         torch.zeros(len(logits)).cuda(), 0])
        # print(logits.shape)
        final_result = []
        final_result_scores = []
        nextid = 0
        tries = 0
        # max_tries = 5000
        num_outs = 0
        curvote = 1
        tmp = args.temperature
        globalsum = 0
        globalnum = 0
        chainids = []
        thisst = 0
        while ((tries < max_tries or num_outs == 0) and tries < 250000):
            currentid = nextid
            tries += 1

            while currentid != -1:
                tc = torch.log(mct_tree[currentid][4])
                tc = tc + F.relu(tc - 10) * 1000
                scoreadd = (mct_tree[currentid][5] + mct_tree[currentid][6]) / mct_tree[currentid][4]

                logits = mct_tree[currentid][0].view(-1) - tc * 0.5 + scoreadd * 0.25
                logits = logits[:50001]
                log_probs = F.softmax(logits, dim=-1)

                pr = torch.multinomial(log_probs, num_samples=1)[0]
                # pr=torch.argmax(logits)
                prev = pr.item()
                # print(logits.shape,currentid,prev)
                # mct_tree[currentid][4][prev]+=1
                lastid = currentid
                chainids.append([currentid, prev])
                currentid = int(mct_tree[currentid][3][prev])

            # start from lastid & currentid

            cqs = mct_tree[lastid][2]
            # print(pr)
            tokens = torch.cat((cqs, pr.unsqueeze(0).view(1, 1)), dim=1)
            output_tokens_list = tokens.view(-1).contiguous()
            # if max_length==min_length:
            #   print(min_length,output_tokens_list,context_length)
            # print(output_tokens_list[context_length:])
            sentence = tokenizer.DecodeIds(output_tokens_list[context_length:].tolist())

            nextid = 0
            ip = checksc(model, tokenizer, args, device, sentence, input)
            if tries % 100 == 0:
                print(input, sentence, ip, tries)
            if ip < -1000:
                # start from scratch again
                if (len(chainids) > 1 and thisst < 500):
                    nextid = chainids[-1][0]
                    chainids = chainids[:-1]
                    thisst += 1
                else:
                    nextid = 0
                    chainids = []
                    thisst = 0
                mct_tree[lastid][4][prev] = 100000
                continue
            thisst = 0
            if len(sentence) == len(input):
                # print(ip,sentence)
                mct_tree[lastid][4][prev] = 100000
                final_result.append([sentence])
                final_result_scores.append(-ip.cpu().item())
                for i in range(len(chainids)):
                    mct_tree[chainids[i][0]][4][chainids[i][1]] += 1
                    mct_tree[chainids[i][0]][5][chainids[i][1]] += ip
                    if i != len(chainids) - 1:
                        mct_tree[chainids[i + 1][0]][6] = mct_tree[chainids[i][0]][5][chainids[i][1]] / \
                                                          mct_tree[chainids[i][0]][4][chainids[i][1]]
                print(input, sentence, ip, tries)
                nextid = 0
                chainids = []
                num_outs += 1
                thisst = 0
                continue

                # calculate
            mct_tree[lastid][3][prev] = len(mct_tree)
            tmp = args.temperature
            # tmp=tmp*(1-0.05*len(sentence))
            rts = mct_tree[lastid][1]
            index = len(tokens[0])

            logits, *rts = model(tokens[:, index - 1: index], tokens.new_ones((1, 1)) * (index - 1),
                                 tokens.new_ones(1, 1, 1, args.mem_length + 1, device=tokens.device,
                                                 dtype=torch.float), *rts)
            logits = logits[0, -1] / tmp

            mct_tree.append([logits, rts, tokens, -np.ones(len(logits)), torch.ones(len(logits)).cuda(),
                             torch.zeros(len(logits)).cuda(), ip])
            # chainids=[]
            nextid = len(mct_tree) - 1
        del mct_tree
        torch.cuda.empty_cache()
        # print(tries,len(final_result))
        return final_result, final_result_scores


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
    args = get_args()
    print(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
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


def prepare_model():
    """Main training program."""

    # print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

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


# def dl_interface(model, tokenizer, args, device, str):
#     pp = pinyin(str, style=TONE3)
#     wq = checkpz(pp[-1][0], str[-1])
#     # 2:出句为上联  1：出句为下联
#     output_string, output_scores = generate_sentence(model, tokenizer, args, device, str)
#     ranklist = np.argsort(output_scores)
#     opt = []
#     for i in range(len(ranklist)):
#         j = ranklist[i]
#         if (output_scores[j] < best_score + 5 and len(opt) < 5):
#             opt.append(output_string[j])
#
#     return wq, opt


def generate_strs(tups):
    model, tokenizer, args = prepare_model()
    output = []
    for tup in tups:
        # str=generate_token_tensor(str,tokenizer)

        output_string, output_scores = generate_sentence(model, tokenizer, args, torch.cuda.current_device(), tup)
        list_poems = 0
        # print(output_string,output_scores)
        ranklist = np.argsort(output_scores)
        best_score = output_scores[ranklist[0]]
        text_dir = "dl_save"
        w = os.listdir()
        if not (text_dir in w):
            os.mkdir(text_dir)
        already = []
        with jsonlines.open(text_dir + '/' + tup + '.jsonl', mode='w') as writer:
            for i in range(len(ranklist)):
                j = ranklist[i]
                if output_scores[j] < best_score + 2:
                    otc = {}

                    otc['chuju'] = tup
                    otc['duiju'] = output_string[j]
                    print(tup, output_string[j], output_scores[j])
                    writer.write(otc)

    return 0


def generate():
    fi = []

    title_list = ["望江楼望江流望江楼上望江流江楼千古江流千古", "越过粤南是越南", "近世进士尽是近视"]
    title_list = ["旧金山新约克不是约克不是金山", "钱钟书轻钱重书", "上海自来水来自海上"]

    for i in title_list:
        fi.append(i)

        # fi.append([i[0],j,i[1]])
    output = generate_strs(fi)


def random_generate(mode=0):
    text_dir = "poems_save_modern3/"
    dist = os.listdir()
    if not (text_dir[:-1] in dist):
        os.mkdir(text_dir[:-1])
    if mode == 0:
        qts = open("selected_102.txt", 'r')

        wt = qts.readlines()
        lt = []
        for i in wt:
            sp = i.split()
            # print(sp)
            author = sp[1]
            title = sp[0]
            num_wd = int(sp[2])
            num_st = int(sp[3]) * 2
            lt.append([author, title, num_wd, num_st])
        qts.close()
    if mode == 1:
        qts = open("index.txt", 'r')

        wt = qts.readlines()
        lt = []
        for i in wt:

            sp = i.split()
            if len(sp) > 0:
                # print(sp)
                author = "李白"
                title = sp[0]
                lt.append([author, title])
        qts.close()

    model, tokenizer, args = prepare_model()
    while True:
        id = random.randint(0, len(lt) - 1)
        # author,title,num_wd,num_st=lt[id]
        author, title = lt[id]
        lists = os.listdir(text_dir)
        lts = title + author + '.jsonl'
        if (lts in lists):
            continue
        # str=generate_token_tensor(str,tokenizer)
        # output_string,output_scores=generate_string(model, tokenizer, args, torch.cuda.current_device(),title,author,length=num_wd)
        output_string, output_scores = generate_string(model, tokenizer, args, torch.cuda.current_device(), title,
                                                       author)
        new_output_string = []
        new_output_score = []
        for i in range(len(output_string)):
            st = output_string[i].replace('。', ',').replace('，', ',').replace('？', ',').replace('?', ',').replace('!',
                                                                                                                  ',').replace(
                '！', ',')
            st = st.split(',')
            # print(st,num_st)
            # if len(st)-1==num_st:
            new_output_string.append(output_string[i])
            new_output_score.append(output_scores[i])
        if len(new_output_string) == 0:
            del output_string
            del output_scores
            continue
        list_poems = 0

        ranklist = np.argsort(new_output_score)
        best_score = new_output_score[ranklist[0]]

        already = []

        with jsonlines.open(text_dir + title + author + '.jsonl', mode='w') as writer:
            for i in range(len(ranklist)):
                j = ranklist[i]
                if new_output_score[j] < best_score + 2:
                    if not (new_output_string[j][0:5] in already):
                        otc = {}
                        otc['author'] = author
                        otc['title'] = title
                        otc['context'] = new_output_string[j]
                        # print(otc)
                        writer.write(otc)
                        already.append(new_output_string[j][0:5])

        del output_string
        del output_scores

    return 0


def dl_interface(model, tokenizer, args, device, content, max_tries):
    pp = pinyin(content, style=TONE3)
    wq = checkpz(pp[-1][0], content[-1])
    # 2:出句为上联  1：出句为下联
    output_string, output_scores = generate_sentence(model, tokenizer, args, device, content, max_tries)
    if output_string == [] and output_scores == []:
        return 0, "请在尝试一次吧"
    ranklist = np.argsort(output_scores)
    best_duiju = output_string[ranklist[0]]
    # opt=[]
    # for i in range(len(ranklist)):
    #     j=ranklist[i]
    #     if (output_scores[j]<best_score+5 and len(opt)<5):
    #         opt.append(output_string[j])

    return wq, best_duiju[0]

# generate()
def generate_duilian(content,model, tokenizer, args,max_tries):
    # pp = pinyin(content, style=TONE3)
    # wq = checkpz(pp[-1][0], content[-1])
    # 2:出句为上联  1：出句为下联
    wq, best_duiju = dl_interface(model, tokenizer, args, torch.cuda.current_device(), content,max_tries)
    # processLogger.info(best_duiju)
    return wq,best_duiju

if __name__ == "__main__":
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["MASTER_PORT"] = "13335"
    # content = "春江花月夜"
    content = "南通州北通州南北通州通南北"
    model, tokenizer, args = prepare_model()
    b = time.time()
    wq,best_duiju = generate_duilian(content,model, tokenizer, args,1000)
    print(time.time() - b)
    print(wq)
    print(best_duiju)

