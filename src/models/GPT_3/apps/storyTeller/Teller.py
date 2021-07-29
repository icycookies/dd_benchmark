import sys
import os
import json
from GPT_3.settings import GENERATE_DIR
sys.path.append(GENERATE_DIR)
print(GENERATE_DIR)
print(sys.path)
from generate_string import *
# from generate_pms2 import *
from generate_pm2_refined import *
from generate_qa_fast import *
from generate_string_fast import *
from generate_nlp_tools import *
from generate_duilia import *
from generate_pms_heading import write_poem_heading
from generate_pms_song import write_poem_song
from fastpoem import fast_poem
from generate_string_refined import news_refined
from generate_string_limit_refined import news_limit_refined

# from generate_jizhong_batch import *
from generate_qa_0127 import *
from generate_qa_noset import *
# from generate_qa_desc import *
from generate_qa_desc_refined import *

from flask import current_app
from GPT_3.settings import ROOT_DIR

class Generator:
    #  冀中小说，甄嬛剧场
    # model_path = '/data/shixw/data/finetune/novel_origin_3cal/txl-2.8b11-20-15-10'
    # 原始写诗 人设问答
    # model_path = '/data/shixw/data/checkpoints_origin/txl-2.8b11-20-15-10'
    # nlp_tools finetune
    #model_path = '/data/shixw/data/finetune/nlp_tools/txl-2.8b11-20-15-10'
    # 邹旭写诗
    # model_path = '/data/shixw/data/checkpoints_pms/txl-2.8b11-20-15-10'

    def __init__(self, model_path='/data/shixw/data/checkpoints_pms/txl-2.8b11-20-15-10'):
        self.model,self.tokenizer,self.args=prepare_model(model_path)
        # current_app.logger.info("test_length： " + str(self.args.out_seq_length))
        #self.output = generate_common(content,self.model,self.tokenizer,self.args)
    def generate(self,content):
        output = generate_common(content,self.model,self.tokenizer,self.args)
        return output
        #return self.output
        #return 1
    def poem(self,content):
        output = write_common(content,self.model,self.tokenizer,self.args)
        return output
    def poem_heading(self, content):
        output = write_poem_heading(content, self.model,self.tokenizer,self.args)
        return output
    def poem_song(self, content):
        output = write_poem_song(content, self.model,self.tokenizer,self.args)
        return output
    def poem_fast(self, content):
        output = fast_poem(content, self.model, self.tokenizer,self.args)
        return output
    def qa(self,content):
        output = qa_common(self.model,self.tokenizer,self.args,content)
        # output = qa_op_0127(self.model,self.tokenizer,self.args,content)
        return output
    # 小剧场专用
    def generate_fast(self,content,req_tag=None,max_chars_count=40):
        if req_tag ==None:
            req_tag = []
        output,text_type = generate_string_fast_fun(self.model,self.tokenizer,self.args,content,req_tag,max_chars_count=max_chars_count)
        return output,text_type
    def generate_simple_fast(self,content,max_length):
        output = generate_fast_clear_fun(self.model,self.tokenizer,self.args,content,max_chars_count=max_length)
        return output
    def generate_news_refined(self, title, desc):
        output = news_refined(self.model, self.tokenizer, self.args, title, desc)
        return output
    def generate_news_limit_refined(self, title, desc):
        output = news_limit_refined(self.model, self.tokenizer, self.args, title, desc)
        return output
    def do_score(self,input_str,eval_str):
        score = generate_score(self.model,self.tokenizer,self.args,None,input_str,eval_str)
        return score
    def source_batch(self,barrage_lines,bg):
        score = source_batch(self.model,self.tokenizer,self.args,barrage_lines,bg)
        return score
    def generate_batch_fast(self,content,batch_size):
        output = batch_generate_fun(self.model,self.tokenizer,self.args,content,batch_size)
        return output
    # 邹旭0303对联
    def generate_duilian_fun(self,content,max_tries):
        wq, best_duiju = generate_duilian(content,self.model,self.tokenizer,self.args,max_tries)
        return wq, best_duiju
    # 邹旭0127优化版
    # 带人设
    def generate_qa_op_0127(self,content):
        output = qa_op_0127(self.model,self.tokenizer,self.args,content)
        return output
    # 不带人设
    def generate_qa_noset_0127(self,content):
        output = qa_noset_0127(self.model,self.tokenizer,self.args,content)
        return output
    # 带desc
    def generate_qa_desc(self, content):
        output = qa_desc(self.model,self.tokenizer,self.args,content)
        return output
    # 文本的关键词，摘要，分类生成
    def generate_kat(self,content):
        output = generate_kws_abs_title(self.model,self.tokenizer,self.args,content)
        return output


if __name__ == "__main__":
    teller = Generator()
    aa = teller.generate_common("咏奥巴马","楚 屈原")
    print(aa)
