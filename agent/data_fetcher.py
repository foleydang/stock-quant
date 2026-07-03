#!/usr/bin/env python3
"""
数据获取层 - 所有外部数据获取的统一入口

职责：从数据库、Tushare、腾讯财经API获取数据，返回结构化dict。
不包含任何飞书/卡片/意图逻辑。
"""

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(os.path.dirname(AGENT_DIR), 'python')
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, PYTHON_DIR)

from config import DB_PATH, TUSHARE_TOKEN, TENCENT_QUOTE_API, WATCHLIST, AVAILABLE_CASH, TOTAL_INVESTMENT

logger = logging.getLogger("feishu_bot")

import time

# ========== 简易内存缓存（减少 Tushare API 调用） ==========
_cache = {}
_CACHE_TTL = 300  # 5分钟过期

def _cache_get(key):
    entry = _cache.get(key)
    if entry and time.time() - entry['time'] < _CACHE_TTL:
        return entry['data']
    return None

def _cache_set(key, data):
    _cache[key] = {'data': data, 'time': time.time()}



# ========== 数据库连接 ==========

def _get_conn():
    return sqlite3.connect(DB_PATH)


# ========== 腾讯财经 API（实时行情） ==========

def _tencent_batch_query(codes: dict) -> dict:
    """批量查询腾讯财经API，返回 {name: {price, change_pct, change_amount}}"""
    import requests
    try:
        query_str = ",".join(codes.values())
        url = f"{TENCENT_QUOTE_API}{query_str}"
        r = requests.get(url, timeout=15)
        lines = r.text.strip().split(";")
        results = {}
        for name, code in codes.items():
            for line in lines:
                if code in line:
                    parts = line.split('~')
                    if len(parts) > 32:
                        results[name] = {
                            'price': float(parts[3]),
                            'change_pct': float(parts[32]),
                            'change_amount': float(parts[31]),
                        }
                    break
        return results
    except Exception as e:
        logger.error(f"腾讯财经API查询异常: {e}")
        return {}


# ========== 持仓数据 ==========

def get_positions_data() -> Dict:
    """获取持仓数据"""
    conn = _get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT symbol, stock_name, shares, cost_price, current_price FROM positions")
        rows = cursor.fetchall()
    except Exception:
        conn.close()
        return {'error': '无法读取持仓数据'}

    positions = []
    total_value = 0
    total_cost = 0

    from data.data_handler import DataHandler
    dh = DataHandler(force_refresh=False)
    symbols = [r[0] for r in rows]
    realtime = dh.get_realtime_prices(symbols) if symbols else {}

    for row in rows:
        symbol, stock_name, shares, cost_price, current_price_db = row[0], row[1], row[2], row[3], row[4] if len(row) > 4 else None
        name = stock_name
        shares = int(shares) if shares else 0
        cost_price = float(cost_price) if cost_price else 0.0
        current_price = realtime.get(symbol, {}).get('price', current_price_db or cost_price)
        current_price = float(current_price) if current_price else 0.0
        market_value = current_price * shares
        cost = cost_price * shares
        profit = market_value - cost
        profit_pct = (current_price - cost_price) / cost_price * 100 if cost_price > 0 else 0
        positions.append({
            'symbol': symbol, 'stock_name': name, 'shares': shares,
            'cost_price': cost_price, 'current_price': current_price,
            'market_value': market_value, 'profit': profit, 'profit_pct': profit_pct,
        })
        total_value += market_value
        total_cost += cost

    conn.close()

    available_cash = AVAILABLE_CASH
    total_profit = total_value - total_cost
    profit_pct = total_profit / total_cost * 100 if total_cost > 0 else 0

    return {
        'positions': positions, 'total_value': total_value + available_cash,
        'total_cost': total_cost + available_cash, 'total_profit': total_profit,
        'profit_pct': profit_pct, 'available_cash': available_cash,
    }


# ========== 个股行情 ==========

def get_stock_data(symbol: str) -> Dict:
    """获取单只股票行情"""
    if not symbol:
        return {'error': '请提供股票代码'}

    from data.data_handler import DataHandler
    dh = DataHandler(force_refresh=False)

    # 优先用实时行情，确保涨跌额和涨跌幅基准一致
    realtime = dh.get_realtime_prices([symbol])
    rt = realtime.get(symbol, {})

    if rt and rt.get('price') and rt.get('prev_close'):
        current_price = rt['price']
        prev_close = rt['prev_close']
        change_amount = current_price - prev_close
        change_pct = (change_amount / prev_close) * 100
    else:
        # 实时行情不可用时，用K线数据
        df = dh.fetch_stock_data(symbol)
        if df is None or df.empty:
            return {'error': f'无法获取 {symbol} 的数据'}
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        current_price = float(latest['close'])
        prev_close = float(prev['close'])
        change_amount = current_price - prev_close
        change_pct = (change_amount / prev_close) * 100

    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
    row = cursor.fetchone()
    name = row[0] if row and row[0] else symbol
    conn.close()

    return {
        'symbol': symbol, 'name': name,
        'current_price': current_price, 'change_pct': change_pct,
        'change_amount': change_amount,
        'volume': rt.get('volume') if rt else None,
    }


# ========== 做T建议 ==========

def get_t_suggestions() -> List[Dict]:
    """获取做T建议"""
    try:
        from lgbm_backtest import LGBMBacktesterOptimized
        positions_data = get_positions_data()
        if 'error' in positions_data:
            return []

        suggestions = []
        for pos in positions_data['positions']:
            if pos.get('profit_pct', 0) > 5:
                bt = LGBMBacktesterOptimized()
                result = bt.run_backtest(pos['symbol'])
                if result and result.get('summary'):
                    suggestions.append({
                        'stock_name': pos['stock_name'], 'symbol': pos['symbol'],
                        'current_price': pos['current_price'],
                        'action': '适合做T' if result['summary']['winRate'] > 50 else '观望',
                        'reason': f"胜率{result['summary']['winRate']:.1f}%",
                    })
        return suggestions[:5]
    except Exception as e:
        logger.error(f"做T建议获取失败: {e}")
        return []


# ========== 交易信号 ==========

def get_signals_data() -> Dict:
    """获取交易信号 - LGBM 模型 window=3 预测均值 + 截面排名信号

    修复：使用全市场截面排名判断信号，而非绝对阈值。
    模型预测值分布偏正 (base ~0.011)，绝对阈值 0.0005 会让所有股票都触发买入。
    改为：Top 30% → 买入，Bottom 30% → 卖出，中间 40% → 持有。
    """
    try:
        from lgbm_backtest import LGBMBacktesterOptimized
        import numpy as np
        import math
        import sqlite3

        bt = LGBMBacktesterOptimized()

        if not bt.models:
            logger.warning("LGBM模型未加载，跳过信号生成")
            return {'signals': []}

        # 从数据库获取更多股票（不只是WATCHLIST），做截面排名
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data/stock_data.db')
        conn = sqlite3.connect(db_path)
        # 获取所有有足够数据的股票（至少200条30分钟K线）
        all_symbols = [row[0] for row in conn.execute(
            "SELECT symbol FROM kline_30m GROUP BY symbol HAVING COUNT(*) >= 200 ORDER BY symbol"
        ).fetchall()]
        # 取前200只做截面（限制计算量）
        all_symbols = all_symbols[:200]

        # 获取股票名称
        name_map = {}
        for row in conn.execute("SELECT symbol, name FROM stock_info").fetchall():
            name_map[row[0]] = row[1]
        conn.close()

        # 加载所有股票数据
        all_data = {}
        for symbol in all_symbols:
            df = bt.load_data(symbol)
            if df is not None and len(df) >= 20:
                df = df.reset_index(drop=True)
                all_data[symbol] = df

        if not all_data:
            logger.warning("LGBM: 无有效股票数据")
            return {'signals': []}

        # 预计算特征
        bt.preload_features(all_data)

        # 窗口参数（回测验证 window=3 最优）
        PRED_WINDOW = 3

        # 对每只股票获取预测
        raw_predictions = []
        for symbol in all_data:
            df = all_data[symbol]
            last_idx = len(df) - 1
            name = name_map.get(symbol, symbol)

            # 取最近 PRED_WINDOW 个 bar 的预测均值
            preds = []
            for j in range(max(0, last_idx - PRED_WINDOW + 1), last_idx + 1):
                p, _ = bt._get_model_prediction(symbol, j)
                preds.append(p)

            avg_pred = float(np.mean(preds)) if preds else 0.0
            pred_std = float(np.std(preds)) if len(preds) > 1 else 0.0
            consistency = sum(1 for p in preds if p > 0) / len(preds) if preds else 0.5

            raw_predictions.append({
                'symbol': symbol, 'name': name,
                'avg_pred': avg_pred, 'pred_std': pred_std,
                'consistency': consistency,
            })

        if not raw_predictions:
            return {'signals': []}

        # 截面排名：基于 avg_pred 的分位数
        all_preds = [p['avg_pred'] for p in raw_predictions]
        p30 = np.percentile(all_preds, 30)
        p70 = np.percentile(all_preds, 70)

        logger.info(f"LGBM截面: N={len(raw_predictions)}, p30={p30:.6f}, p70={p70:.6f}, "
                    f"min={min(all_preds):.6f}, max={max(all_preds):.6f}")

        # 生成信号
        signals = []
        for p in raw_predictions:
            avg_pred = p['avg_pred']
            consistency = p['consistency']

            # 将预期收益率映射为上涨概率（使用sigmoid）
            up_prob = 1.0 / (1.0 + math.exp(-avg_pred * 500))
            up_prob = max(0.0, min(1.0, up_prob))

            # 获取当前价格
            rt = get_stock_data(p['symbol'])
            current_price = rt.get('current_price', 0) if rt else 0

            # 截面排名信号：Top 30% → 买入，Bottom 30% → 卖出
            if avg_pred >= p70:
                action = '买入'
            elif avg_pred <= p30:
                action = '卖出'
            else:
                action = '持有'

            signals.append({
                'stock_name': p['name'], 'symbol': p['symbol'],
                'current_price': current_price,
                'signal': action, 'up_prob': up_prob,
                'reason': f"预期收益{avg_pred:+.4f}(σ={p['pred_std']:.4f},c={consistency:.0%}) | 概率{up_prob:.1%}",
            })

        # 按概率排序
        signals.sort(key=lambda x: x['up_prob'], reverse=True)

        buy_count = sum(1 for s in signals if s['signal'] == '买入')
        sell_count = sum(1 for s in signals if s['signal'] == '卖出')
        logger.info(f"LGBM信号: {len(signals)}只 (买入{buy_count}/卖出{sell_count}/持有{len(signals)-buy_count-sell_count})")

        return {'signals': signals}
    except Exception as e:
        logger.error(f"信号获取失败: {e}")
        import traceback
        traceback.print_exc()
        return {'signals': []}


# ========== 回测 ==========

def run_backtest(symbol: str) -> Dict:
    """运行LGBM回测"""
    try:
        from lgbm_backtest import LGBMBacktesterOptimized
        bt = LGBMBacktesterOptimized()
        result = bt.run_backtest(symbol)
        if not result:
            return {'error': f'{symbol} 回测失败'}
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
        row = cursor.fetchone()
        result['symbol'] = symbol
        result['name'] = row[0] if row else symbol
        conn.close()
        return result
    except Exception as e:
        return {'error': f'回测异常: {str(e)[:50]}'}


# ========== 盘后总结 ==========

def get_daily_summary() -> Dict:
    """获取盘后总结"""
    positions_data = get_positions_data()
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_value': positions_data.get('total_value', 0),
        'total_profit': positions_data.get('total_profit', 0),
        'profit_pct': positions_data.get('profit_pct', 0),
        'positions': positions_data.get('positions', []),
    }


# ========== 综合分析 ==========

def analyze_stock(symbol: str) -> Dict:
    """综合分析个股"""
    return get_stock_data(symbol)


# ========== 自选股管理 ==========

def manage_watchlist(action: str, symbol: str, name: str = '') -> Dict:
    """管理自选股（自动持久化到 config.yaml）"""
    global WATCHLIST
    if action == 'add':
        for w in WATCHLIST:
            if w.get('symbol') == symbol:
                return {'message': f'{name}({symbol}) 已在自选列表'}
        WATCHLIST.append({'symbol': symbol, 'name': name})
        from config import save_watchlist
        save_watchlist()
        return {'message': f'✓ 已添加 {name}({symbol}) 到自选列表'}
    elif action == 'remove':
        for i, w in enumerate(WATCHLIST):
            if w.get('symbol') == symbol:
                WATCHLIST.pop(i)
                return {'message': f'✓ 已移除 {name}({symbol})'}
        return {'message': f'{symbol} 不在自选列表'}
    return {'error': '未知操作'}


# ========== 大盘指数（腾讯财经API） ==========

def get_market_data() -> Dict:
    cached = _cache_get('market_data')
    if cached:
        return cached
    indices_config = [
        {"name": "上证指数", "code": "sh000001", "display_code": "000001.SH"},
        {"name": "深证成指", "code": "sz399001", "display_code": "399001.SZ"},
        {"name": "创业板指", "code": "sz399006", "display_code": "399006.SZ"},
        {"name": "沪深300", "code": "sh000300", "display_code": "000300.SH"},
        {"name": "恒生指数", "code": "r_hkHSI", "display_code": "HSI"},
    ]

    results = _tencent_batch_query({cfg["name"]: cfg["code"] for cfg in indices_config})
    indices = []
    for cfg in indices_config:
        r = results.get(cfg["name"])
        if r:
            indices.append({"name": cfg["name"], "code": cfg["display_code"], "price": r["price"], "change_pct": r["change_pct"], "change_amount": r["change_amount"]})

    avg = sum(i.get('change_pct', 0) for i in indices) / len(indices) if indices else 0
    sentiment = "🔴 市场偏强" if avg > 1 else "🟢 市场偏弱" if avg < -1 else "⚪ 市场震荡"
    result = {"indices": indices, "sentiment": sentiment}
    _cache_set("market_data", result)
    return result


# ========== 行业板块（腾讯财经API） ==========

def get_sector_data() -> Dict:
    cached = _cache_get('sector_data')
    if cached:
        return cached
    sector_codes = {
        "食品饮料": "sh000036", "银行": "sh000022", "房地产": "sh000021",
        "医药生物": "sh000020", "电子": "sh000018", "计算机": "sh000017",
        "机械设备": "sh000019", "有色金属": "sh000033", "化工": "sh000034",
        "汽车": "sh000029", "家用电器": "sh000030", "非银金融": "sh000028",
        "公用事业": "sh000027", "通信": "sh000026", "传媒": "sh000025",
        "交通运输": "sh000023", "农林牧渔": "sh000014", "钢铁": "sh000032",
    }

    results = _tencent_batch_query(sector_codes)
    sectors = [{"name": name, "change_pct": r["change_pct"], "price": r["price"], "lead_stock": ""} for name, r in results.items()]
    sectors.sort(key=lambda s: s['change_pct'], reverse=True)
    result = {"sectors": sectors[:10]}
    _cache_set("sector_data", result)
    return result


# ========== 多股对比 ==========

def compare_stocks(symbols: list) -> Dict:
    stocks = [get_stock_data(s) for s in symbols[:5]]
    stocks = [s for s in stocks if 'error' not in s]
    return {"stocks": stocks, "count": len(stocks)} if stocks else {"error": "无法获取对比数据"}

# ========== 资金流向 ==========

def get_money_flow(symbol: str) -> Dict:
    cached = _cache_get(f'money_flow_{symbol}')
    if cached:
        return cached
    """获取个股资金流向（Tushare moneyflow）"""
    try:
        import tushare as ts
        pro = ts.pro_api(TUSHARE_TOKEN)
        # 获取最近5天的资金流向
        from datetime import datetime, timedelta
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        df = pro.moneyflow(ts_code=symbol, start_date=start, end_date=end)
        if df is None or df.empty:
            return {'error': f'{symbol} 资金流向数据暂无'}

        latest = df.iloc[-1]
        net_mf = latest.get('net_mf_amount', 0)  # 万元
        buy_lg = latest.get('buy_lg_amount', 0)
        sell_lg = latest.get('sell_lg_amount', 0)
        buy_sm = latest.get('buy_sm_amount', 0)
        sell_sm = latest.get('sell_sm_amount', 0)
        
        conn = _get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM stock_info WHERE symbol=?", (symbol,))
        row = cursor.fetchone()
        name = row[0] if row else symbol
        conn.close()

        # 最近5天趋势
        trend = []
        for _, r in df.tail(5).iterrows():
            trend.append({
                'date': str(r.get('trade_date', '')),
                'net_mf': r.get('net_mf_amount', 0),
            })

        return {
            'symbol': symbol, 'name': name,
            'net_mf_amount': float(net_mf),  # 万元
            'buy_lg_amount': float(buy_lg), 'sell_lg_amount': float(sell_lg),
            'buy_sm_amount': float(buy_sm), 'sell_sm_amount': float(sell_sm),
            'lg_net': float(buy_lg) - float(sell_lg),  # 大单净流入
            'sm_net': float(buy_sm) - float(sell_sm),  # 小单净流入
            'date': str(latest.get('trade_date', '')),
            'trend': trend,
        }
    except Exception as e:
        logger.error(f"资金流向获取失败: {e}")
        return {'error': f'资金流向获取异常: {str(e)[:50]}'}


# ========== 个股深度数据 ==========

def get_stock_deep_data(symbol: str) -> Dict:
    cached = _cache_get(f'deep_data_{symbol}')
    if cached:
        return cached
    """获取个股深度数据（估值+盈利+量价）"""
    try:
        import tushare as ts
        pro = ts.pro_api(TUSHARE_TOKEN)

        # 1. 估值数据
        from datetime import datetime, timedelta
        trade_date = datetime.now().strftime('%Y%m%d')
        basic = pro.daily_basic(ts_code=symbol, trade_date=trade_date,
                                fields='ts_code,trade_date,close,pe,pe_ttm,pb,ps,ps_ttm,total_mv,circ_mv,turnover_rate,volume_ratio')
        
        valuation = {}
        if basic is not None and not basic.empty:
            r = basic.iloc[-1]
            valuation = {
                'pe': r.get('pe'), 'pe_ttm': r.get('pe_ttm'),
                'pb': r.get('pb'), 'ps': r.get('ps'),
                'total_mv': r.get('total_mv'),  # 万元
                'turnover_rate': r.get('turnover_rate'),
                'volume_ratio': r.get('volume_ratio'),
            }

        # 如果当天没有数据，取最近有数据的
        if not valuation or all(v is None for v in valuation.values()):
            for i in range(1, 10):
                td = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
                basic = pro.daily_basic(ts_code=symbol, trade_date=td,
                                        fields='ts_code,trade_date,close,pe,pe_ttm,pb,total_mv,turnover_rate')
                if basic is not None and not basic.empty:
                    r = basic.iloc[-1]
                    valuation = {
                        'pe': r.get('pe'), 'pe_ttm': r.get('pe_ttm'),
                        'pb': r.get('pb'), 'total_mv': r.get('total_mv'),
                        'turnover_rate': r.get('turnover_rate'),
                        'date': str(r.get('trade_date', '')),
                    }
                    break

        # 2. 盈利数据
        fina = pro.fina_indicator(ts_code=symbol, start_date='20250101', end_date='20260601',
                                  fields='ts_code,ann_date,end_date,roe,roe_waa,netprofit_margin,grossprofit_margin,or_yoy,netprofit_yoy')
        
        profit = {}
        if fina is not None and not fina.empty:
            r = fina.iloc[-1]
            profit = {
                'roe': r.get('roe'), 'netprofit_margin': r.get('netprofit_margin'),
                'grossprofit_margin': r.get('grossprofit_margin'),
                'or_yoy': r.get('or_yoy'),  # 营收增速
                'netprofit_yoy': r.get('netprofit_yoy'),  # 净利润增速
                'end_date': str(r.get('end_date', '')),
            }

        # 3. 实时行情
        stock = get_stock_data(symbol)
        name = stock.get('name', symbol)
        current_price = stock.get('current_price', 0)
        change_pct = stock.get('change_pct', 0)

        return {
            'symbol': symbol, 'name': name,
            'current_price': current_price, 'change_pct': change_pct,
            'valuation': valuation, 'profit': profit,
        }
    except Exception as e:
        logger.error(f"深度数据获取失败: {e}")
        return {'error': f'深度数据异常: {str(e)[:50]}'}


# ========== 对比增强 ==========

def compare_stocks_deep(symbols: list) -> Dict:
    """增强版对比：价格+估值+盈利"""
    stocks = []
    for s in symbols[:5]:
        try:
            deep = get_stock_deep_data(s)
            if 'error' not in deep:
                stocks.append(deep)
        except Exception:
            pass

    if not stocks:
        return {'error': '无法获取对比数据'}

    # 计算估值排名
    ranked_by_pe = sorted(stocks, key=lambda s: s.get('valuation', {}).get('pe_ttm', 0) or 999)
    
    return {
        'stocks': stocks, 'count': len(stocks),
        'cheapest': ranked_by_pe[0].get('name', '') if ranked_by_pe else '',
    }


# ========== 北向资金 ==========

def get_north_flow() -> Dict:
    """获取北向资金流向（沪股通+深股通合并）"""
    try:
        import tushare as ts
        pro = ts.pro_api(TUSHARE_TOKEN)
        from datetime import datetime, timedelta
        
        # 尝试最近3个交易日（避免非交易日无数据）
        for offset in range(1, 4):
            trade_date = (datetime.now() - timedelta(days=offset)).strftime('%Y%m%d')
            df_sh = pro.hsgt_top10(trade_date=trade_date, market_type='1')  # 1=沪股通
            df_sz = pro.hsgt_top10(trade_date=trade_date, market_type='2')  # 2=深股通
            
            if (df_sh is not None and not df_sh.empty) or (df_sz is not None and not df_sz.empty):
                break
        
        results = []
        
        # 沪股通 top5
        if df_sh is not None and not df_sh.empty:
            for _, r in df_sh.head(5).iterrows():
                results.append({
                    'name': str(r.get('name', '')),
                    'symbol': str(r.get('ts_code', '')),
                    'net_buy': float(r.get('net_buy', 0)),  # 万元
                    'close': float(r.get('close', 0)),
                    'change_pct': float(r.get('change', 0)),
                    'channel': '沪股通',
                })
        
        # 深股通 top5
        if df_sz is not None and not df_sz.empty:
            for _, r in df_sz.head(5).iterrows():
                results.append({
                    'name': str(r.get('name', '')),
                    'symbol': str(r.get('ts_code', '')),
                    'net_buy': float(r.get('net_buy', 0)),
                    'close': float(r.get('close', 0)),
                    'change_pct': float(r.get('change', 0)),
                    'channel': '深股通',
                })
        
        # 按净买入排序，取前10
        results.sort(key=lambda x: x.get('net_buy', 0), reverse=True)
        results = results[:10]
        
        total_net = sum(r.get('net_buy', 0) for r in results) if results else 0
        
        return {'stocks': results, 'total_net': total_net, 'date': trade_date}
    except Exception as e:
        logger.error(f"北向资金获取失败: {e}")
        return {'error': '北向资金获取异常'}
