from com_utils.http_utils import HttpUtil
from flask import request, Blueprint

__all__ = ['test']

test = Blueprint('test', __name__)

@test.route('/do/<string:content>', methods=['GET'])
def do_test(content):
    try:
        data = content
        return HttpUtil.http_response(0, 'success', data)
    except Exception as e:
        return HttpUtil.http_response(1, 'failed')
