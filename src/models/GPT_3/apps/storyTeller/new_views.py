# -*- coding:utf-8 -*-
from com_utils.http_utils import HttpUtil
from flask import request, Blueprint
from GPT_3.apps.storyTeller.views import generator
from GPT_3.apps.storyTeller.tools import *
from GPT_3 import app
import time
import logging
import traceback
import requests
import re
import random
from flask import current_app
from GPT_3.settings import BASE_DIR


__all__ = ['qa', 'poetry', 'news']
processLogger = logging.getLogger('process')
qa = Blueprint('qa', __name__)
poetry = Blueprint('poetry', __name__)
news = Blueprint('news', __name__)
joint = "&&"


@qa.route('/normal/', methods=['POST'])
def qa_normal():
    try:
        question = HttpUtil.check_param("question", request, method=1)
        desc = HttpUtil.check_param("desc", request, method=1)
        processLogger.info("question: {}  desc:{}".format(question, desc))
        content = joint.join([question, desc])
        story = generator.generate_qa_desc(content)
        processLogger.info("story: {}".format(story))
        return HttpUtil.http_response(0, 'success', story)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)


@news.route('/generate/', methods=['POST'])
def news_generate():
    try:
        content = HttpUtil.check_param("content", request, method=1)
        max_length = HttpUtil.check_param("max_length", request, required=False, method=1, default=100)
        processLogger.info("content: {}  max_length: {}".format(content, max_length))
        story = generator.generate_simple_fast(content, max_length)
        processLogger.info("story: {}".format(story))
        return HttpUtil.http_response(0, 'success', story)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)


@news.route('/refined/', methods=['POST'])
def news_refined():
    try:
        title = HttpUtil.check_param("title", request, method=1)
        desc = HttpUtil.check_param("desc", request, method=1)
        processLogger.info("title: {}  desc: {}".format(title, desc))
        story = generator.generate_news_refined(title, desc)
        processLogger.info("story: {}".format(story))
        return HttpUtil.http_response(0, 'success', story)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)


@news.route('/limit_refined/', methods=['POST'])
def news_limit_refined():
    try:
        title = HttpUtil.check_param("title", request, method=1)
        desc = HttpUtil.check_param("desc", request, method=1)
        processLogger.info("title: {}  desc: {}".format(title, desc))
        story = generator.generate_news_limit_refined(title, desc)
        processLogger.info("story: {}".format(story))
        return HttpUtil.http_response(0, 'success', story)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)


@poetry.route('/normal/', methods=["POST"])
def poetry_normal():
    try:
        title = HttpUtil.check_param("title", request, method=1)
        author = HttpUtil.check_param("author", request, method=1)
        desc = HttpUtil.check_param("desc", request, method=1)
        content = joint.join([title, author, desc])
        processLogger.info("写诗请求："+ content)
        data = generator.poem(content)
        processLogger.info("写诗返回："  + data + "--请求：" + content)
        return HttpUtil.http_response(0, 'success', data)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)


@poetry.route('/fast/', methods=["POST"])
def poetry_fast():
    try:
        title = HttpUtil.check_param("title", request, method=1)
        author = HttpUtil.check_param("author", request, method=1)
        desc = HttpUtil.check_param("desc", request, method=1)
        content = joint.join([title, author, desc])
        processLogger.info("写诗请求："+ content)
        data = generator.poem_fast(content)
        processLogger.info("写诗返回："  + data + "--请求：" + content)
        return HttpUtil.http_response(0, 'success', data)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)


@poetry.route('/heading/', methods=["POST"])
def poetry_heading():
    try:
        title = HttpUtil.check_param("title", request, method=1)
        author = HttpUtil.check_param("author", request, method=1)
        desc = HttpUtil.check_param("desc", request, method=1)
        heading = HttpUtil.check_param("heading", request, method=1)
        content = joint.join([title, author, desc, heading])
        processLogger.info("写诗请求："+ content)
        data = generator.poem_heading(content)
        processLogger.info("写诗返回："  + data + "--请求：" + content)
        return HttpUtil.http_response(0, 'success', data)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)


@poetry.route('/songci/', methods=["POST"])
def poetry_songci():
    try:
        title = HttpUtil.check_param("title", request, method=1)
        author = HttpUtil.check_param("author", request, method=1)
        desc = HttpUtil.check_param("desc", request, method=1)
        cipai = HttpUtil.check_param("cipai", request, method=1)
        content = joint.join([title, author, desc, cipai])
        processLogger.info("写诗请求："+ content)
        data = generator.poem_song(content)
        processLogger.info("写诗返回："  + data + "--请求：" + content)
        return HttpUtil.http_response(0, 'success', data)
    except Exception as e:
        processLogger.error(traceback.format_exc())
        return HttpUtil.http_response(1, 'failed', e)
