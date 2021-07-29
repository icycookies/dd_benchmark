# from com_utils.http_utils import HttpUtil
# from flask import request, Blueprint
# from GPT_3.apps.roleSettingQA.RoleSetQA import RoleSetQA
# from flask import current_app
#
# __all__ = ['roleQA']
#
# roleQA = Blueprint('roleQA', __name__)
#
# roleSetQA = RoleSetQA()
# @roleQA.route('/answer/', methods=['POST'])
# def roleASetQA_view():
#     try:
#         content = HttpUtil.check_param("content", request, method=1)
#         print("**收到请求"+content)
#         story = roleSetQA.answer_fun(content)
#         return HttpUtil.http_response(0, 'success', story)
#     except Exception as e:
#         return HttpUtil.http_response(1, 'failed',e)
#
