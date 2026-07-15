"""A股/港股/ETF 交易成本模型。

回测(按收益率分数)与纸面引擎(按成交金额)共用同一套费率常量,
避免两处口径漂移导致拼接曲线出现人为拐点。

A股规则(2023-08 起):
  佣金   commission = max(amount × 万2.5, 5元),买卖双边
  印花税 stamp      = amount × 0.05%,仅卖出
  过户费 transfer   = amount × 0.001%,买卖双边(沪深现均收)
  ETF: 免印花税 + 免过户费,只有佣金。
港股: 印花税 0.1% 双边 + 佣金(粗略,港股持仓少)。
"""

# ---- A股费率 ----
COMMISSION_RATE = 0.00025   # 万2.5
COMMISSION_MIN = 5.0        # 单笔佣金下限(元)
STAMP_RATE = 0.0005         # 印花税,仅卖出,ETF 免
TRANSFER_RATE = 0.00001     # 过户费,双边,ETF 免

# ---- 港股费率(粗略)----
HK_STAMP_RATE = 0.001       # 双边
HK_COMMISSION_RATE = 0.00025

# ETF 代码前缀(.SH/.SZ):51/56/58 沪,15 深
_ETF_PREFIXES = ('51', '56', '58', '15')


def is_etf(symbol: str) -> bool:
    """按代码判定是否 A股 ETF(免印花税+过户费)。"""
    if not symbol:
        return False
    s = symbol.upper()
    if not (s.endswith('.SH') or s.endswith('.SZ')):
        return False
    code = s.split('.')[0]
    return code.startswith(_ETF_PREFIXES)


def a_share_cost(amount: float, side: str, etf: bool = False) -> float:
    """A股/ETF 单笔成交的绝对成本(元)。amount=成交金额,side='buy'|'sell'。"""
    amount = abs(float(amount))
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    stamp = amount * STAMP_RATE if (side == 'sell' and not etf) else 0.0
    transfer = amount * TRANSFER_RATE if not etf else 0.0
    return commission + stamp + transfer


def hk_cost(amount: float, side: str) -> float:
    """港股单笔成交的绝对成本(元,粗略)。"""
    amount = abs(float(amount))
    return amount * HK_COMMISSION_RATE + amount * HK_STAMP_RATE


def trade_cost(symbol: str, amount: float, side: str) -> float:
    """按 symbol 后缀自动路由的单笔绝对成本(元)。"""
    s = (symbol or '').upper()
    if s.endswith('.HK'):
        return hk_cost(amount, side)
    return a_share_cost(amount, side, etf=is_etf(s))


def roundtrip_frac(etf: bool = False) -> float:
    """一次买入+一次卖出的成本占成交金额的比例(忽略 5 元佣金下限)。

    供回测使用:回测按收益率分数记账,无逐笔金额,故用费率分数近似。
    A股个股 ≈ 0.00102(0.102%);ETF ≈ 0.0005(0.05%)。
    """
    buy = COMMISSION_RATE + (0.0 if etf else TRANSFER_RATE)
    sell = COMMISSION_RATE + (0.0 if etf else STAMP_RATE + TRANSFER_RATE)
    return buy + sell


def hk_roundtrip_frac() -> float:
    """港股一次往返成本分数(粗略)= 2×佣金 + 2×印花税。"""
    return 2 * HK_COMMISSION_RATE + 2 * HK_STAMP_RATE


def apply_slippage(price: float, side: str, bps: float = 0.0) -> float:
    """开盘价成交,默认 0 滑点(个人低频)。bps>0 时买加卖减。"""
    if not bps:
        return float(price)
    adj = price * (bps / 1e4)
    return float(price + adj) if side == 'buy' else float(price - adj)


if __name__ == '__main__':
    import sys
    # 手算对照自检
    def chk(name, got, want):
        ok = abs(got - want) < 1e-6
        print(f"{'✓' if ok else '✗'} {name}: got={got:.5f} want={want:.5f}")
        return ok

    all_ok = True
    # A股买入 10万: 佣金 100000*0.00025=25 > 5, 过户 100000*0.00001=1, 印花0 → 26
    all_ok &= chk('A股买10万', a_share_cost(100000, 'buy', etf=False), 26.0)
    # A股卖出 10万: 佣金25 + 印花 100000*0.0005=50 + 过户1 → 76
    all_ok &= chk('A股卖10万', a_share_cost(100000, 'sell', etf=False), 76.0)
    # 小额买 1万: 佣金 10000*0.00025=2.5 < 5 → 取5, 过户 0.1 → 5.1
    all_ok &= chk('A股买1万(佣金下限)', a_share_cost(10000, 'buy', etf=False), 5.1)
    # ETF卖 10万: 仅佣金25, 免印花免过户 → 25
    all_ok &= chk('ETF卖10万', a_share_cost(100000, 'sell', etf=True), 25.0)
    # 港股卖 10万: 佣金25 + 印花100 → 125
    all_ok &= chk('港股卖10万', hk_cost(100000, 'sell'), 125.0)
    # 往返分数: 个股 0.00025+0.00001 + 0.00025+0.0005+0.00001 = 0.00102
    all_ok &= chk('个股往返分数', roundtrip_frac(etf=False), 0.00102)
    # ETF 往返分数: 0.00025 + 0.00025 = 0.0005
    all_ok &= chk('ETF往返分数', roundtrip_frac(etf=True), 0.0005)
    # trade_cost 路由
    all_ok &= chk('trade_cost ETF(159792.SZ)卖', trade_cost('159792.SZ', 100000, 'sell'), 25.0)
    all_ok &= chk('trade_cost 个股(600048.SH)卖', trade_cost('600048.SH', 100000, 'sell'), 76.0)
    all_ok &= chk('trade_cost 港股(3690.HK)卖', trade_cost('3690.HK', 100000, 'sell'), 125.0)
    print('\n全部通过 ✓' if all_ok else '\n有失败 ✗')
    sys.exit(0 if all_ok else 1)
