#!/usr/bin/env python3
"""
HTTP 服务层 - 飞书事件回调入口

职责：接收飞书 HTTP POST → 分发事件 → 回复结果
只做 HTTP 通信，不做任何业务逻辑。

运行方式：gunicorn start_bot_http:app
"""

import json
import logging
import os
import sys
import traceback
import threading

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(BASE_DIR, 'agent')
PYTHON_DIR = os.path.join(BASE_DIR, 'python')

# 加载 .env
env_file = os.path.join(AGENT_DIR, '.env')
with open(env_file) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip()

sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, PYTHON_DIR)

from config import FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_VERIFICATION_TOKEN, FEISHU_ENCRYPT_KEY, BOT_PORT

# 日志
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout), FlushingFileHandler(os.path.join(LOG_DIR, 'feishu_bot.log'), encoding='utf-8')],
)
logger = logging.getLogger("feishu_bot")

# ===== SDK monkey-patch =====
from lark_oapi.api.im.v1 import P2ImMessageReceiveV1
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler, EventException

_orig = EventDispatcherHandler._do_without_validation
def _safe(self, payload: bytes):
    try:
        return _orig(self, payload)
    except EventException as e:
        logger.info(f"[SAFE] 忽略未知事件: {e}")
        return None
EventDispatcherHandler._do_without_validation = _safe
logger.info("✓ SDK monkey-patch 已生效")

# ===== 消息回调 =====
def on_message_receive(data: P2ImMessageReceiveV1) -> None:
    event = data.event
    message = event.message
    sender = event.sender

    if sender.sender_type == "APP":
        logger.info("忽略机器人自身消息")
        return

    msg_type = message.message_type
    try:
        content = json.loads(message.content)
        text = content.get("text", "").strip()
    except json.JSONDecodeError:
        text = message.content

    if not text or msg_type != "text":
        return

    message_id = message.message_id
    chat_id = message.chat_id
    chat_type = message.chat_type
    logger.info(f"✓✓ 收到飞书消息: chat_id={chat_id}, chat_type={chat_type}, text={text}")

    def _process():
        try:
            from bot_server import process_message
            from feishu_client import reply_card
            card = process_message(text)
            if card:
                card_str = json.dumps(card, ensure_ascii=False)
                logger.info(f"✓✓ 回复卡片: {card_str[:300]}")
                result = reply_card(message_id, card)
                logger.info("✓✓ 已回复卡片消息" if result else "✗✗ 回复失败")
        except Exception as e:
            logger.error(f"✗✗ 处理异常: {e}")

    threading.Thread(target=_process, daemon=True).start()


def catch_all(data) -> None:
    try:
        event_type = data.header.event_type if hasattr(data, 'header') and hasattr(data.header, 'event_type') else 'unknown'
        logger.info(f"事件: {event_type}")
    except Exception:
        pass

# ===== 事件处理器 =====
handler = EventDispatcherHandler.builder(FEISHU_ENCRYPT_KEY, FEISHU_VERIFICATION_TOKEN) \
    .register_p2_im_message_receive_v1(on_message_receive) \
    .register_p2_im_message_message_read_v1(catch_all) \
    .build()

# ===== 定时推送 =====
from scheduler import start_scheduler
start_scheduler()

# ===== Flask HTTP 服务 =====
from flask import Flask, request
from lark_oapi.core.model import RawRequest
from lark_oapi.core.const import UTF_8

app = Flask(__name__)

@app.route('/feishu/event', methods=['POST'])
def event_callback():
    try:
        body = request.get_data(as_text=True)
        logger.info(f"收到 HTTP 事件回调: {body[:200]}")
        raw_req = RawRequest()
        raw_req.uri = request.path
        raw_req.headers = dict(request.headers)
        raw_req.body = body.encode(UTF_8)
        result = handler.do(raw_req)
        logger.info(f"SDK 处理结果: status={result.status_code}")
        if result.content:
            return result.content.decode(UTF_8)
        return json.dumps({'code': 0})
    except Exception as e:
        logger.error(f"事件回调异常: {e}")
        return json.dumps({'code': -1, 'msg': str(e)})


@app.route('/health', methods=['GET'])
def health():
    return {"status": "ok", "mode": "http", "port": BOT_PORT}


# ===== 启动日志 =====
logger.info(f"金融小助手启动 - HTTP 回调模式")
logger.info(f"飞书 APP_ID: {FEISHU_APP_ID}")
logger.info(f"回调地址: https://stock.yanten.top/feishu/event")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=BOT_PORT, debug=False)