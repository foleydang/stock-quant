#!/usr/bin/env python3
"""
飞书消息发送客户端
使用 lark-oapi SDK 发送消息和卡片
"""

import json
import os
import logging

from lark_oapi.api.im.v1 import *
from lark_oapi import Client, LogLevel

import sys, os
# 确保 import agent 的 config，不是 python 的
_agent_dir = os.path.dirname(os.path.abspath(__file__))
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

from config import FEISHU_APP_ID, FEISHU_APP_SECRET

import traceback

logger = logging.getLogger(__name__)

# 创建飞书 Client
_client = None


def get_client():
    """获取飞书 API Client"""
    global _client
    if _client is None:
        if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
            raise ValueError("飞书 APP_ID / APP_SECRET 未配置，请设置环境变量")
        _client = Client.builder() \
            .app_id(FEISHU_APP_ID) \
            .app_secret(FEISHU_APP_SECRET) \
            .log_level(LogLevel.WARNING) \
            .build()
    return _client


def send_text(chat_id: str, text: str) -> bool:
    """发送纯文本消息"""
    try:
        client = get_client()
        req = CreateMessageRequest.builder() \
            .receive_id_type("chat_id") \
            .request_body(CreateMessageRequestBody.builder()
                          .receive_id(chat_id)
                          .msg_type("text")
                          .content(json.dumps({"text": text}))
                          .build()) \
            .build()

        resp = client.im.v1.message.create(req)
        if not resp.success():
            logger.error(f"发送文本消息失败: {resp.code} {resp.msg}")
            return False
        return True
    except Exception as e:
        logger.error(f"发送文本消息异常: {e}", exc_info=True)
        return False


def send_card(chat_id: str, card: dict) -> bool:
    """发送交互式消息卡片"""
    try:
        client = get_client()
        req = CreateMessageRequest.builder() \
            .receive_id_type("chat_id") \
            .request_body(CreateMessageRequestBody.builder()
                          .receive_id(chat_id)
                          .msg_type("interactive")
                          .content(json.dumps(card))
                          .build()) \
            .build()

        resp = client.im.v1.message.create(req)
        if not resp.success():
            logger.error(f"发送卡片消息失败: {resp.code} {resp.msg}")
            return False
        logger.info(f"✓ 卡片已发送到 {chat_id}")
        return True
    except Exception as e:
        logger.error(f"发送卡片消息异常: {e}", exc_info=True)
        return False


def reply_text(message_id: str, text: str) -> bool:
    """回复文本消息"""
    try:
        client = get_client()
        req = ReplyMessageRequest.builder() \
            .request_body(ReplyMessageRequestBody.builder()
                          .msg_type("text")
                          .content(json.dumps({"text": text}))
                          .build()) \
            .message_id(message_id) \
            .build()

        resp = client.im.v1.message.reply(req)
        if not resp.success():
            logger.error(f"回复消息失败: {resp.code} {resp.msg}")
            return False
        return True
    except Exception as e:
        logger.error(f"回复消息异常: {e}", exc_info=True)
        return False


def reply_card(message_id: str, card: dict) -> bool:
    """回复卡片消息"""
    try:
        client = get_client()
        req = ReplyMessageRequest.builder() \
            .request_body(ReplyMessageRequestBody.builder()
                          .msg_type("interactive")
                          .content(json.dumps(card))
                          .build()) \
            .message_id(message_id) \
            .build()

        resp = client.im.v1.message.reply(req)
        if not resp.success():
            logger.error(f"回复卡片失败: {resp.code} {resp.msg}")
            return False
        return True
    except Exception as e:
        logger.error(f"回复卡片异常: {e}", exc_info=True)
        return False


def send_card_to_user(open_id: str, card: dict) -> bool:
    """发送卡片给个人用户"""
    try:
        client = get_client()
        req = CreateMessageRequest.builder() \
            .receive_id_type("open_id") \
            .request_body(CreateMessageRequestBody.builder()
                          .receive_id(open_id)
                          .msg_type("interactive")
                          .content(json.dumps(card))
                          .build()) \
            .build()

        resp = client.im.v1.message.create(req)
        if not resp.success():
            logger.error(f"发送卡片给用户失败: {resp.code} {resp.msg}")
            return False
        return True
    except Exception as e:
        logger.error(f"发送卡片给用户异常: {e}", exc_info=True)
        return False