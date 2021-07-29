# -*- coding:utf-8 -*-
from com_utils.http_utils import HttpUtil
from flask import request, Blueprint
from GPT_3.apps.storyTeller.Teller import Generator
from GPT_3.apps.storyTeller.ScoreIndicator import ScoreIndicator
from GPT_3.apps.storyTeller.tools import *
from GPT_3 import app
import time
import logging
import traceback
import requests
import re
import random
processLogger = logging.getLogger('process')
from flask import current_app
from GPT_3.settings import BASE_DIR

__all__ = ['storyTeller','write_poem','write_poem_heading','qa_view','write_couplet']

model_path = app.config.get("MODEL_PATH")
print("model_path: ", model_path)
storyTeller = Blueprint('storyTeller', __name__)
generator = Generator(model_path)
scoreIndicator = ScoreIndicator()
request_lock = False

#加载nsfc学科类别
code2field_nsfc = {}
nsfc_path = BASE_DIR + "/apps/storyTeller/script/subject_nsfc.json"
f = open(nsfc_path)
d = json.load(f)
for k,v in d.items():
    c2f_list = k.split(":")
    code2field_nsfc[c2f_list[0]] = c2f_list[1]
    for item in v:
        if type(v) == dict:
            for k_s,v_s in v.items():
                c2f_list_s = k_s.split(":")
                code2field_nsfc[c2f_list_s[0]] = c2f_list_s[1]
                for v_s_s in v_s:
                    c2f_list_s_s = v_s_s.split(":")
                    code2field_nsfc[c2f_list_s_s[0]] = c2f_list_s_s[1]
        else:
            c2f_list_s = item.split(":")
            code2field_nsfc[c2f_list_s[0]] = c2f_list_s[1]


ZH_PATTERN = re.compile(u'[\u4e00-\u9fa5]+')
EN_PATTERN = re.compile('[a-zA-Z]')
def contain_zh(word):
    global ZH_PATTERN
    match = re.search(ZH_PATTERN, word)
    if match:
        return True
    else:
        return False

@storyTeller.route('/tell/', methods=['POST'])
def tell():
    global request_lock
    try:
        content = HttpUtil.check_param("content", request, method=1)
        print("**收到请求"+content)
        story = generator.generate(content)
        return HttpUtil.http_response(0, 'success', story)
    except Exception as e:
        return HttpUtil.http_response(1, 'failed', e)



@storyTeller.route('/qa/', methods=['POST'])
def tell_qa():
    try:
        user_id = ""
        story = ""
        text_type = 0
        content = HttpUtil.check_param("content", request, method=1)
        # quality_level 0:2秒速度版 5:甄嬛反向推理筛选 6:甄嬛问答格式fintune 7:甄嬛小说finetune 8:甄嬛小说截断生成finetune
        quality_level = HttpUtil.check_param("quality_level", request, required=False, method=1, default=0)
        barrage_list = HttpUtil.check_param("bullet_list", request, required=False, method=1, default=[])
        req_tag = HttpUtil.check_param("req_tag", request, required=False, method=1, default=[])
        processLogger.info("content: {}  quality_level:{}".format(content,quality_level))
        role_times = 3
        # 先找出当前角色
        role_line = content[-6:]
        if barrage_list:
            try:
                story,user_id = scoreIndicator.get_good_bullet(barrage_list[:30])
                if not story:
                    # 如果背景输入中有'第'或者'集'字
                    #content = change_section(content)
                    while role_times:
                        role_times -= 1
                        if quality_level == 5:
                            story, text_type = get_good_output(content,role_line)
                            break
                        elif quality_level == 6:
                            story, text_type = generator.generate_fast(content,max_chars_count=250)
                        elif quality_level == 7:
                            story, text_type = generator.generate_fast(content,max_chars_count=600)
                        elif quality_level == 8:
                            story, text_type = generator.generate_fast(content,max_chars_count=600)
                            story = auto_trunc(story)
                        else:
                            story, text_type = generator.generate_fast(content)
                            if text_type==2 or zhenhuang_roleCla(story,role_line):
                                break
                text_type = 1
            except Exception:
                processLogger.error(traceback.format_exc())
                return HttpUtil.http_response(0, 'success', {"content": barrage_list[0]["content"], "bullet_id": barrage_list[0]["user_id"]})
                # return HttpUtil.http_response(0, 'success', {"content": "", "bullet_id": "ee","type":text_type})
        else:
            # 2 秒
            if quality_level == 0:
                content = change_section(content)
                while role_times:
                    role_times -= 1
                    story,text_type = generator.generate_fast(content,req_tag)
                    if text_type==2 or zhenhuang_roleCla(story, role_line):
                        break
                if text_type != 2:
                    scoreIndicator.change_score(story)
            # 6: 甄嬛问答格式fintune
            elif quality_level == 6:
                story, text_type = generator.generate_fast(content, max_chars_count=250)
            # 7:甄嬛小说finetune
            elif quality_level == 7:
                story, text_type = generator.generate_fast(content, max_chars_count=600)
            # 8:甄嬛小说截断生成finetune
            elif quality_level == 8:
                story, text_type = generator.generate_fast(content, max_chars_count=600)
                story = auto_trunc(story)
            # 反向推理筛选
            elif quality_level == 5:
                story, text_type = get_good_output(content,role_line)
                scoreIndicator.change_score(story)
            # 优化版 慢 无人设语音问答
            elif quality_level == 1:
                story = generator.generate_qa_noset_0127(content)
            # 优化版 慢 人设问答
            elif quality_level == 2:
                story = generator.generate_qa_op_0127(content)
        processLogger.info("story: {}".format(story))

        return HttpUtil.http_response(0, 'success', {"content":story,"bullet_id":user_id,"type":text_type})
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)

@storyTeller.route('/kws_abs_title_tools/', methods=['POST'])
def kws_abs_title_tools():
    try:
        text = HttpUtil.check_param("content", request, method=1)
        if not text:
            raise Exception("content is null")
        processLogger.info("content: {} ".format(text))
        content = "摘要:"+text+" 关键词:"
        res = generator.generate_kat(content)
        processLogger.info("story: {}".format(res))
        kws = []
        title =""
        field = ""
        field_code = ""
        if res:
            # 关键词
            kws_line = res[res.rfind("关键词:")+4:res.find("领域:")]
            if kws_line:
                if "#" in kws_line:
                    kws = kws_line.split("#")
                else:
                    kws = kws_line.split(" ")
                kws = [item.strip() for item in kws]
            # 领域
            field_code = res[res.rfind("领域:")+3:res.find("标题:")].strip()
            field = code2field_nsfc.get(field_code,"")
            # 摘要
            title = res[res.rfind("标题:")+3:]
        processLogger.info("kws :{}, field:{} ,field_code {},abstract:{}".format(kws,field,field_code,text))
        return HttpUtil.http_response(0, 'success', {"content": text, "kws": kws, "field_code":field_code,"field":field,"title": title})
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)


@storyTeller.route('/getGoodBarrage/', methods=['POST'])
def score_view():
    try:
        current_app.logger.info("get a 请求:")
        content = HttpUtil.check_param("content", request, method=1)
        barrage_list = HttpUtil.check_param("barrage_list", request, method=1)
        print(content)
        print(barrage_list)
        res = generator.generate_fast(content)
        next_sentence = res
        end_token = [".", "。", "！", "!", "?", "？", "；", ";"]
        for index,c in enumerate(res):
            if c in end_token:
                next_sentence = res[:index+1]
                break
        barrage_score = {}
        for barrage in barrage_list:
            print(barrage)
            this_score = generator.do_score(barrage,next_sentence)
            print(this_score)
            barrage_score[barrage] = this_score
            print('--------------------------------------')

        sort_barrage = sorted(barrage_score.items(),key=lambda x:x[1],reverse=True)
        print()
        high_barrage = sort_barrage[0][0]
        print("high_barrage:",high_barrage)



        return high_barrage
    except Exception as e:
        print(e)
        current_app.logger.info("error： " + str(e))
        return HttpUtil.http_response(1, 'failed', e)

@storyTeller.route('/batchGetGoodBarrage/', methods=['POST'])
def batch_score_view():
    try:
        current_app.logger.info("get a 请求:")
        background = HttpUtil.check_param("background", request, method=1)
        barrage_list = HttpUtil.check_param("barrage_list", request, method=1)
        good_barrage = generator.source_batch(barrage_list,background)
        return good_barrage
    except Exception as e:
        print(e)
        processLogger.error(traceback.format_exc())
        current_app.logger.info("error： " + str(e))
        return HttpUtil.http_response(1, 'failed', e)

@storyTeller.route('/', methods=['POST'])
def qa_view():
    # 人设问答
    try:
        content = HttpUtil.check_param("content", request, method=1)
        quality_level = HttpUtil.check_param("quality_level", request, required=False, method=1, default=0)
        processLogger.info("content: {}  quality_level:{}".format(content, quality_level))
        # 2 秒 问答
        if quality_level == 0:
            story = generator.qa(content)
        # 优化版 慢 语音问答
        elif quality_level == 1:
            story = generator.generate_qa_noset_0127(content)
        # 优化版 慢 人设问答
        elif quality_level == 2:
            story = generator.generate_qa_op_0127(content)
        # 带描述的问答
        elif quality_level == 10:
            story = generator.generate_qa_desc(content)
        
        processLogger.info("story: {}".format(story))
        return HttpUtil.http_response(0, 'success', story)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)
# 快速 不加任何加工的 最开始为孟茜 服务
@storyTeller.route('/simple_generate/', methods=['POST'])
def generate_mq_view():
    try:
        content = HttpUtil.check_param("content", request, method=1)
        max_length = HttpUtil.check_param("max_length", request, required=False, method=1, default=100)
        processLogger.info("req : {}. content: {}  ".format("mq", content))
        story = generator.generate_simple_fast(content,max_length)
        processLogger.info("story: {}".format(story))
        return HttpUtil.http_response(0, 'success', story)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)


@storyTeller.route('/write_poem/', methods=['POST'])
def write_poem():
    try:
        content = HttpUtil.check_param("content", request, method=1)
        processLogger.info("写诗请求："+ content)
        story = generator.poem(content)
        data = story
        end = time.time()
        # time_consume = end - begin
        processLogger.info("写诗返回："  + data + "--请求：" + content)
        return HttpUtil.http_response(0, 'success', data)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)

@storyTeller.route('/write_poem_heading/', methods=['POST'])
def write_poem_heading():
    try:
        content = HttpUtil.check_param("content", request, method=1)
        processLogger.info("写诗请求："+ content)
        story = generator.poem_heading(content)
        data = story
        end = time.time()
        # time_consume = end - begin
        processLogger.info("写诗返回："  + data + "--请求：" + content)
        return HttpUtil.http_response(0, 'success', data)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)

@storyTeller.route('/write_poem_song/', methods=['POST'])
def write_poem_song():
    try:
        content = HttpUtil.check_param("content", request, method=1)
        processLogger.info("写诗请求："+ content)
        story = generator.poem_song(content)
        data = story
        end = time.time()
        # time_consume = end - begin
        processLogger.info("写诗返回："  + data + "--请求：" + content)
        return HttpUtil.http_response(0, 'success', data)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)

@storyTeller.route('/write_poem_fast/', methods=['POST'])
def write_poem_fast():
    try:
        content = HttpUtil.check_param("content", request, method=1)
        processLogger.info("写诗请求："+ content)
        story = generator.poem_fast(content)
        data = story
        end = time.time()
        # time_consume = end - begin
        processLogger.info("写诗返回："  + data + "--请求：" + content)
        return HttpUtil.http_response(0, 'success', data)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)

# @storyTeller.route('/write_poem/', methods=['POST'])
# def write_poem():
#     global request_lock
#     try:
#         content = HttpUtil.check_param("content", request, method=1)
#         current_app.logger.info("写诗请求："+ content)
#         b = time.time()
#         while True:
#             lock.acquire()
#             if not request_lock:
#                 request_lock = True
#                 story = teller.poem(content)
#                 request_lock = False
#                 lock.release()
#                 data = story
#                 current_app.logger.info("写诗返回："  + data + "--请求：" + content)
#                 return HttpUtil.http_response(0, 'success', data)
#             else:
#                 e = time.time()
#                 if (e - b) > 1000:
#                     lock.release()
#                     raise Exception("访问人太多")
#             lock.release()
#     except Exception as e:
#         if lock.acquire():
#             lock.release()
#         return HttpUtil.http_response(1, 'failed', e)

#邹旭对联
@storyTeller.route('/couplet/', methods=['POST'])
def write_couplet():
    try:
        content = HttpUtil.check_param("content", request, required=True,method=1)
        max_tries = HttpUtil.check_param("max_tries", request, required=False, method=1,default=5000)
        if max_tries > 5000 or max_tries < 1000:
            raise Exception('max_tries must in range [1000,5000]')
        processLogger.info("chuju : {}  max_tries:{}".format(content,max_tries))
        wq, best_duiju = generator.generate_duilian_fun(content,max_tries)
        data = {"pos":wq,"result":best_duiju}
        # data = json.dumps(data, indent=4, ensure_ascii=False)
        processLogger.info("chuju: {},duiju: {} ,pos :{}".format(content,best_duiju,wq))
        return HttpUtil.http_response(0, 'success', data)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)
# @storyTeller.route('/couplet/<string:content>', methods=['GET'])
# def write_couplet(content):
#     try:
#         max_tries = HttpUtil.check_param("max_tries", request, required=False, default=5000)
#         if max_tries > 5000 or max_tries < 1000:
#             raise Exception('max_tries must in range [1000,5000]')
#         processLogger.info("chuju : {}  max_tries:{}".format(content,max_tries))
#         wq, best_duiju = generator.generate_duilian_fun(content,max_tries)
#         data = {"pos":wq,"result":best_duiju}
#         # data = json.dumps(data, indent=4, ensure_ascii=False)
#         processLogger.info("chuju: {},duiju: {} ,pos :{}".format(content,best_duiju,wq))
#         return HttpUtil.http_response(0, 'success', data)
#     except Exception as e:
#         processLogger.error(traceback.format_exc())
#         return HttpUtil.http_response(1, 'failed', e)
