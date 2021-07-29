import time
import os

filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)),"sensitive_words.txt")
# DFA算法
class DFAFilter(object):
    def __init__(self):
        self.keyword_chains = {}  # 关键词链表
        self.delimit = '\x00'  # 限定
        with open(filePath, encoding='utf-8') as f:
            for keyword in f:
                self.add(str(keyword).strip())

    def add(self, keyword):
        keyword = keyword.lower()  # 关键词英文变为小写
        chars = keyword.strip()  # 关键字去除首尾空格和换行
        if not chars:  # 如果关键词为空直接返回
            return
        level = self.keyword_chains
        # 遍历关键字的每个字
        for i in range(len(chars)):
            # 如果这个字已经存在字符链的key中就进入其子字典
            if chars[i] in level:
                level = level[chars[i]]
            else:
                if not isinstance(level, dict):
                    break
                for j in range(i, len(chars)):
                    level[chars[j]] = {}
                    last_level, last_char = level, chars[j]
                    level = level[chars[j]]
                last_level[last_char] = {self.delimit: 0}
                break
        if i == len(chars) - 1:
            level[self.delimit] = 0

    # def parse(self, path):
    #     with open(path, encoding='utf-8') as f:
    #         for keyword in f:
    #             self.add(str(keyword).strip())
    #     print(self.keyword_chains)

    def filter(self, message, repl="*"):
        message = message.lower()
        ret = []
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            for char in message[start:]:
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:
                        level = level[char]
                    else:
                        ret.append(repl * step_ins)
                        start += step_ins - 1
                        break
                else:
                    ret.append(message[start])
                    break
            else:
                ret.append(message[start])
            start += 1

        return ''.join(ret)


    def filter_fun(self, message):
        message = message.lower()
        start = 0
        while start < len(message):
            level = self.keyword_chains
            step_ins = 0
            for char in message[start:]:
                if char in level:
                    step_ins += 1
                    if self.delimit not in level[char]:
                        level = level[char]
                    else:
                        start += step_ins - 1
                        return True
                else:
                    break
            start += 1
        return False

gfw = DFAFilter()


if __name__ == "__main__":
    # gfw = DFAFilter()
    # gfw.parse(path)
    text = "雍正元年，结束了血腥的夺位之争，新的君主继位，国泰民安，政治清明，大傻子但在一片祥和的表象之下，一股暗流蠢蠢欲动，尤其后宫，华妃与皇后分庭抗礼，各方势力裹挟其中，凶险异常。这天皇上宫中遇到华妃。"
    time1 = time.time()
    # result = gfw.filter(text)
    result = gfw.filter_fun(text)

    print(text)
    print(result)
    time2 = time.time()
    print('总共耗时：' + str(time2 - time1) + 's')