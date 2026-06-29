"""
东方财富 Choice 量化 API 接口

等待客户经理开通后使用：
1. 客户经理会给你账号和密码/Token
2. 下载 SDK: https://quantapi.eastmoney.com/Download/Download
3. 安装: pip install ChoiceSDK_xxx.whl (或客户经理给的包名)
4. 在此文件填入你的账号信息

官方文档: https://quantapi.eastmoney.com/
"""

# TODO: 客户经理开通后填入你的账号信息
CHOICE_USERNAME = ''  # Choice 账号
CHOICE_PASSWORD = ''  # Choice 密码 或 Token

_choice_available = False
_choice_api = None

def init_choice():
    """初始化 Choice API"""
    global _choice_available, _choice_api
    
    if not CHOICE_USERNAME or not CHOICE_PASSWORD:
        return False
    
    try:
        # Choice SDK 导入（安装后才能用）
        from ChoiceAPI import Choice  # 或客户经理告诉的包名
        
        # 登录
        _choice_api = Choice()
        _choice_api.login(CHOICE_USERNAME, CHOICE_PASSWORD)
        _choice_available = True
        return True
    except ImportError:
        print("Choice SDK 未安装，请从官网下载: https://quantapi.eastmoney.com/")
        return False
    except Exception as e:
        print(f"Choice 登录失败: {e}")
        return False


def fetch_kline_30m(symbol: str, days: int = 60):
    """
    获取 30 分钟 K 线数据
    
    Args:
        symbol: 股票代码如 '600036.SH'
        days: 获取天数
    
    Returns:
        DataFrame 或 None
    """
    if not _choice_available:
        return None
    
    try:
        # Choice API 获取分钟数据
        # 具体函数名等客户经理开通后确认
        df = _choice_api.get_kline(
            code=symbol,
            cycle='30m',
            start_date='-60d',
            end_date='today'
        )
        
        # 转换格式
        df = df.rename(columns={
            'time': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    except Exception as e:
        print(f"Choice 获取数据失败: {e}")
        return None


# 检查可用性
if init_choice():
    print("✅ Choice API 已初始化")
else:
    print("⚠️ Choice API 未配置，等待客户经理开通")
