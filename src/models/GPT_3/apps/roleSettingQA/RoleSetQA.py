# import sys
# import os
# import json
# from GPT_3.settings import GENERATE_DIR
# sys.path.append(GENERATE_DIR)
# print(GENERATE_DIR)
# print(sys.path)
# from generate_string import *
# # from generate_pms2 import *
#
# class RoleSetQA:
#
#     def __init__(self):
#         self.model,self.tokenizer,self.args=prepare_model(160000)
#         #self.output = generate_common(content,self.model,self.tokenizer,self.args)
#     def answer_fun(self,content):
#         output = generate_common(content,self.model,self.tokenizer,self.args)
#         return output
#
#
#
#
#
# if __name__ == "__main__":
#     roleSetQA = RoleSetQA()
#     aa = roleSetQA.answer("咏奥巴马","楚 屈原")
#     print(aa)
