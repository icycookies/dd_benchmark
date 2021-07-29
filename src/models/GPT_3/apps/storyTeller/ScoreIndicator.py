import json
import os
import jieba
import re
import time
import logging
from GPT_3.settings import BASE_DIR
processLogger = logging.getLogger('process')
ZH_PATTERN = re.compile(u'[\u4e00-\u9fa5]+')
EN_PATTERN = re.compile('[a-zA-Z]')
def contain_zh(word):
    global ZH_PATTERN
    match = re.search(ZH_PATTERN, word)
    if match:
        return True
    else:
        return False

word_score_path = BASE_DIR + "/apps/storyTeller/script/words_score_table.json"
word2score_diy_path = BASE_DIR + "/apps/storyTeller/script/word2score_diy.json"
stop_word_path = BASE_DIR+"/apps/storyTeller/script/stopword.json"
# 甄嬛小说 弹幕评分
class ScoreIndicator():
    def __init__(self):
        self.threadhold = 100
        self.max_score = 1000
        self.all_table = self.load_words_score_table(word_score_path)
        self.table = {}
        self.set_table()
        processLogger.info("threadhold: {} ".format(self.threadhold))
        self.diy_table = self.load_diy_words_score_table(word2score_diy_path)
        self.stopword = self.load_stopword(stop_word_path)
        self.begin_time = time.time()
    # 重新更新关键词分数表
    def set_table(self):
        sorted_score = sorted(self.all_table.items(), key=lambda x: x[1], reverse=True)
        tmp_table = sorted_score[:int(len(sorted_score) / 2)]
        for item in tmp_table:
            self.table[item[0]] = item[1]
        this_threadhold = int(tmp_table[-1][1])
        this_max_score = sorted_score[0][1]
        if this_threadhold > self.threadhold:
            self.threadhold = this_threadhold
        if this_max_score > self.max_score:
            self.max_score = this_max_score

    # 加载停用词
    def load_stopword(self,stop_word_path):
        f = open(stop_word_path)
        stopword = json.load(f)
        return stopword

    # 加载word分数表
    def load_words_score_table(self,word_score_path):
        if not os.path.exists(word_score_path):
            this_words_score_table = {}
        else:
            f = open(word_score_path)
            this_words_score_table = json.load(f)
            tmp_table = {}
            for item in list(this_words_score_table.keys()):
                if len(item) >= 2 or item in ["朕","臣","妾","君","嬛"]:
                    tmp_table[item] = this_words_score_table[item]
            this_words_score_table = tmp_table
        return this_words_score_table
    # 加载diy word分数表
    def load_diy_words_score_table(self,word2score_diy_path):
        if not os.path.exists(word2score_diy_path):
            word2score_diy_table = {}
        else:
            f = open(word2score_diy_path)
            word2score_diy_table = json.load(f)
        return word2score_diy_table
    # 修改分数表 和 阈值
    def change_score(self,sentence):
        cha = time.time() - self.begin_time
        # 如果超过1小时就更新
        if cha > 3600:
            # 重新更新关键词分数表
            self.set_table()
            # 保存词分数表
            self.save()
            # 修改最新时间
            self.begin_time = time.time()
        if len(sentence)<=5 or not contain_zh(sentence):
            return 0
        words_list = jieba.lcut(sentence)
        for word in words_list:
            if word not in self.stopword and contain_zh(word):
                self.all_table.setdefault(word,0)
                self.all_table[word]+=1
        return 1
    # 定期保存词分数表
    def save(self):
        data = json.dumps(self.all_table, indent=4, ensure_ascii=False)
        with open(word_score_path, 'w', encoding='utf-8') as fw:
            fw.write(data)
        return
    # 筛选最佳句子
    def get_good_bullet(self,bullet_list):
        bullet2score = []
        for item in bullet_list:
            diy_flag = False
            sentence = item["content"]
            this_score = 0
            score_count = 0
            # 先检查是否是diy中的弹幕
            for diy_word in list(self.diy_table.keys()):
                if diy_word in sentence:
                    this_score += 2*self.max_score
                    diy_flag = True
                    break
            if len(sentence) > 5 or contain_zh(sentence):
                words_list = jieba.lcut(sentence)
                for word in words_list:
                    if word not in self.stopword and contain_zh(word):
                        # processLogger.info("*************")
                        # processLogger.info(sentence)
                        # processLogger.info(word)
                        origin_score = this_score
                        this_score += self.table.get(word,0)
                        if this_score > origin_score:
                            score_count += 1
            if ((score_count >= 2 or diy_flag) and this_score > self.threadhold) or ((score_count >= 1 or diy_flag) and this_score > self.threadhold+200) or score_count >= 3:
                # processLogger.info("{} :{}".format(sentence,this_score))
                bullet2score.append({"user_id":item["bullet_id"],"bullet":sentence,"score":this_score})
        if bullet2score:
            sort_bullet = sorted(bullet2score,key=lambda x:x["score"],reverse=True)
            return sort_bullet[0]["bullet"], sort_bullet[0]["user_id"]
        else:
            return "",""

        # print("----------------------------")
        # print(self.table)
        # print("****************************")
        # print(sort_bullet)



if __name__ == "__main__":
    scoreIndicator = ScoreIndicator()
    sentence_list = ["事已至此,母亲何不做个顺水人情","正派的，主流的，正统的。雍正元年，结束了血腥的夺位之争"]
    for s in sentence_list:
        scoreIndicator.change_score(s)
        print(scoreIndicator.table)
    bullet_list = [
        {
            "user_id": "test_user_id_001",
            "content": "好了？。我累了"
        },
        {
            "user_id": "test_user_id_001",
            "content": "好了？。好的。四爷昨进宫了，你可见着 了？"
        },
        {
            "user_id": "test_user_id_001",
            "content": "好了？。四爷夺位了？"
        },
        {
            "user_id": "test_user_id_001",
            "content": "好了？。一直重复？"
        }
    ]

    print(scoreIndicator.get_good_bullet(bullet_list))
    # get_stopword()