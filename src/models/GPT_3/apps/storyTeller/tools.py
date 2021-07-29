# -*- coding:utf-8 -*-
import requests
import json
import re
import random
import copy
import logging
import traceback
import random


processLogger = logging.getLogger('process')
# from GPT_3.apps.storyTeller.views import generator

# 小剧场 随机选集
def change_section(content):
    if "第" in content and "集" in content:
        try:
            # 先将<!--替换成，普通字符l
            content = content.replace("第", "l")
            # 再将-->替换成，普通字符l
            content = content.replace("集", "l")
            # 分组标定，替换，
            pattern = re.compile(r'(l)(.*)(l)')
            new_str = "第{}集".format(random.randint(1, 50))
            # 如果想包括两个l，则用pattern.sub(r\1''\3,Content)
            new_content = (pattern.sub(new_str, content))
        except Exception:
            new_content = content
    else:
        new_content = content
    return new_content

# bert 句子相似度
def requests_for_sim_fun(sentence_1,sentence_2):
    try:
        url = "http://127.0.0.1:19550/BERT/sentence_similar/"
        gpt_res = requests.post(url, json={"sentence_1":sentence_1,"sentence_2":sentence_2}, timeout=60).json()
        if gpt_res["status"] == 1:
            return 0
        else:
            return gpt_res["result"]
    except Exception:
        return 0
# 角色分类  角色列别判断
role2cla = {
    "玄凌":"皇上",
    "甄嬛":"妃子",
    "华妃":"妃子",
}
def zhenhuang_roleCla(text,current_role):
    end_token = [".", "。", "！", "!", "?", "？", "；", ";"]

    for r in role2cla:
        if r in current_role:
           ob_cla =  role2cla[r]
    text = get_first_sentence(text)
    try:
        url = "http://127.0.0.1:19550/role_cla/zhenhuan_role/"
        gpt_res = requests.post(url, json={"text": text}, timeout=6).json()
        processLogger.info('text:{}  role : {}'.format(text,ob_cla))
        processLogger.info(gpt_res["result"])
        if gpt_res["status"] == 1:
            return 0
        else:
            if gpt_res["result"]:
                if gpt_res["result"][ob_cla] >= 0.2:
                    return True
                else:
                    return False
    except Exception:
        return 0


# 获取第一句
def get_first_sentence(text):
    end_token = [".", "。", "！", "!", "?", "？", "；", ";"]
    for index,c in enumerate(text):
        if c in end_token:
            text = text[:index+1]
            break
    return text


# 截断
def auto_trunc(text):
    try:
        r = random.randint(2, 3)
        end_token = [",","，", "。", "！", "!", "?", "？", "；", ";",":","："]
        b = len(text)-1
        for i in range(len(text) - 1, -1, -1):
            if text[i] in end_token and len(text[i:b]) > 10:
                ii = i + int(len(text[i+1:b])/r)
                return text[:ii]
        return text
    except Exception:
        return text





# 找得分最高的输出，反向推理
def get_good_output(content,role_line,loops_num=20):
    from GPT_3.apps.storyTeller.views import generator
    text_type = 0
    content_list = []
    times = 2*loops_num
    while times:
        times -= 1
        try:
            story, text_type = generator.generate_fast(content)

            if zhenhuang_roleCla(story, role_line):
                new_dict = {"text":story}
                content_list.append(new_dict)
                processLogger.info("story: {}   role_line:".format(story, role_line))
                if len(content_list) >= loops_num:
                    break
        except Exception as e:
            processLogger.info('-----------------------')
            processLogger.error(traceback.format_exc())
            continue
    processLogger.info("content_list: {}".format(content_list))
    if not content_list:
        new_dict = {"text": story}
        content_list.append(new_dict)
    tmp_content_list = copy.deepcopy(content_list)
    for index,content_item in enumerate(tmp_content_list):
        first_sentence = get_first_sentence(content_item["text"])
        input = '“' +  first_sentence + '”'+'这句，回答了问题：'
        this_score = generator.do_score(input, content)
        processLogger.info('************************')
        processLogger.info('{}/{}'.format(index,len(content_list)))
        processLogger.info("content: {}   score: {}".format(content_item["text"],this_score))
        content_list[index]['score'] = this_score
    sort_story = sorted(content_list, key=lambda x: x["score"], reverse=True)

    return sort_story[0]['text'],text_type





