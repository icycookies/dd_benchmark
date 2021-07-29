import requests
import logging
import traceback

processLogger = logging.getLogger('process')

def sensitive_fun(sentence):
    # swf
    # POST https://libs.aminer.cn/swf/
    try:
        response = requests.post(
            url="https://libs.aminer.cn/swf/",
            headers={"Content-Type": "text/plain; charset=utf-8",},
            json={"content":sentence}).json()
        return response["hit"]
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return True
if __name__ == "__main__":
    sentence = "雍正元年，结束了血腥的夺位之争，新的君主继位，国泰民安，政治清明，但在一片祥和的表象之下，一股暗流蠢蠢 欲动，尤其后宫，华妃与皇后分庭抗礼，各方势力裹挟其中，凶险异常。这天皇上宫中遇到华妃。本来以她的得宠，仇家应该是少不得她几分，但她却摆脱了昔日卑微的受宠身份，在这里与皇上 迎面相遇，真是叫人刮目相看。"
    # sentence = "雍正元年，"
    print(sensitive_fun(sentence))