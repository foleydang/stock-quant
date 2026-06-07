#!/usr/bin/env python3
"""
飞书金融小助手 - HTTP 回调模式启动

飞书通过 HTTP POST 推送事件到本服务（https://stock.yanten.top/feishu/event）。
不再使用 WebSocket 链接模式，HTTP 回调更稳定可靠。

依赖：Flask (HTTP 服务) + gunicorn (生产部署)
"""

import os
import sys
import json
import logging
import traceback
import threading

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载 .env
env_file = os.path.join(BASE_DIR, 'agent', '.env')
with open(env_file) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

sys.path.insert(0, os.path.join(BASE_DIR, 'agent'))
sys.path.insert(0, os.path.join(BASE_DIR, 'python'))

FEISHU_APP_ID = os.environ.get('FEISHU_APP_ID', '')
FEISHU_APP_SECRET = os.environ.get('FEISHU_APP_SECRET', '')
FEISHU_VERIFICATION_TOKEN = os.environ.get('FEISHU_VERIFICATION_TOKEN', '')
FEISHU_ENCRYPT_KEY = os.environ.get('FEISHU_ENCRYPT_KEY', '')
BOT_PORT = int(os.environ.get('BOT_PORT', '8001'))

# 日志 - 强制 flush
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        FlushingFileHandler(os.path.join(LOG_DIR, 'feishu_bot.log'), encoding='utf-8', mode='a'),
    ]
)
logger = logging.getLogger("feishu_bot")

from lark_oapi.api.im.v1 import P2ImMessageReceiveV1
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler, EventException

# ===== Monkey-patch: 让 _do_without_validation 不抛异常 =====
_original_do_without_validation = EventDispatcherHandler._do_without_validation

def _safe_do_without_validation(self, payload: bytes):
    try:
        return _original_do_without_validation(self, payload)
    except EventException as e:
        logger.info(f"[SAFE] 忽略未知事件: {e}")
        return None

EventDispatcherHandler._do_without_validation = _safe_do_without_validation
logger.info("✓ SDK monkey-patch 已生效")

# ===== 消息处理器 =====

def on_message_receive(data: P2ImMessageReceiveV1) -> None:
    """收到飞书消息回调 - 在线程中处理"""
    try:
        event = data.event
        message = event.message
        sender = event.sender

        if sender.sender_type == "APP":
            logger.info("忽略机器人自身消息")
            return

        msg_type = message.message_type
        content_str = message.content
        try:
            content = json.loads(content_str)
            text = content.get("text", "").strip()
        except json.JSONDecodeError:
            text = content_str

        if not text or msg_type != "text":
            logger.info(f"忽略非文本消息: type={msg_type}")
            return

        message_id = message.message_id
        chat_id = message.chat_id
        chat_type = message.chat_type
        logger.info(f"✓✓ 收到飞书消息: chat_id={chat_id}, chat_type={chat_type}, text={text}")

        def _process_and_reply():
            try:
                from bot_server import process_message
                from feishu_client import reply_card
                card = process_message(text)
                if card:
                    # 记录卡片回复内容（前300字符）
                    card_str = json.dumps(card, ensure_ascii=False)
                    logger.info(f"✓✓ 回复卡片内容: {card_str[:300]}")
                    result = reply_card(message_id, card)
                    if result:
                        logger.info("✓✓ 已回复卡片消息")
                    else:
                        logger.warning("✗✗ 回复卡片失败")
                else:
                    logger.warning("处理消息返回空卡片")
            except Exception as e:
                logger.error(f"✗✗ 处理消息异常: {e}")
                logger.error(traceback.format_exc())

        t = threading.Thread(target=_process_and_reply, daemon=True)
        t.start()

    except Exception as e:
        logger.error(f"✗✗ 解析消息异常: {e}")
        logger.error(traceback.format_exc())


def catch_all_event(data) -> None:
    try:
        event_type = data.header.event_type if hasattr(data, 'header') and hasattr(data.header, 'event_type') else 'unknown'
        logger.info(f"事件: {event_type}")
    except Exception as e:
        logger.error(f"catch_all异常: {e}")


# ===== 构建事件处理器 =====
# 注意：SDK builder 参数顺序是 (encrypt_key, verification_token)
# encrypt_key 为空时跳过签名验证
handler = EventDispatcherHandler.builder(FEISHU_ENCRYPT_KEY, FEISHU_VERIFICATION_TOKEN) \
    .register_p2_im_message_receive_v1(on_message_receive) \
    .register_p2_im_message_message_read_v1(catch_all_event) \
    .build()

# ===== 启动定时推送 =====
from bot_server import start_scheduler_if_configured
start_scheduler_if_configured()

# ===== HTTP 服务 (接收飞书事件回调) =====
from flask import Flask, request
from lark_oapi.core.model import RawRequest
from lark_oapi.core.const import APPLICATION_JSON, UTF_8

app = Flask(__name__)

@app.route('/feishu/event', methods=['POST'])
def event_callback():
    """飞书事件回调入口"""
    try:
        body = request.get_data(as_text=True)
        logger.info(f"收到 HTTP 事件回调: {body[:200]}")

        # 构建 SDK RawRequest
        raw_req = RawRequest()
        raw_req.uri = request.path
        raw_req.headers = dict(request.headers)
        raw_req.body = body.encode(UTF_8)

        # 用 SDK 处理事件（自动处理 challenge、token验证、事件分发）
        result = handler.do(raw_req)
        logger.info(f"SDK 处理结果: status={result.status_code}")

        if result.content:
            return result.content.decode(UTF_8)
        return json.dumps({'code': 0})

    except Exception as e:
        logger.error(f"事件回调处理异常: {e}")
        logger.error(traceback.format_exc())
        return json.dumps({'code': -1, 'msg': str(e)})


@app.route('/health', methods=['GET'])
def health():
    return {"status": "ok", "mode": "http", "port": BOT_PORT}

# ===== 启动 =====
logger.info(f"金融小助手启动 - HTTP 回调模式")
logger.info(f"飞书 APP_ID: {FEISHU_APP_ID}")
logger.info(f"回调地址: https://stock.yanten.top/feishu/event")
logger.info(f"健康检查: http://0.0.0.0:{BOT_PORT}/health")

# gunicorn 模式：不调用 app.run()，gunicorn 会自动加载 app
# 直接运行模式：手动启动 Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=BOT_PORT, debug=False)