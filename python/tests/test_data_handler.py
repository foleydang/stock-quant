"""
数据处理器测试
"""
import unittest
import sys
sys.path.insert(0, '..')

from data.data_handler import DataHandler

class TestDataHandler(unittest.TestCase):
    """数据处理器测试"""
    
    def setUp(self):
        self.handler = DataHandler()
    
    def test_get_realtime_price_a_stock(self):
        """测试A股实时价格获取"""
        prices = self.handler.get_realtime_prices(['300124.SZ'])
        self.assertIsInstance(prices, dict)
        if '300124.SZ' in prices:
            self.assertGreater(prices['300124.SZ']['price'], 0)
    
    def test_get_realtime_price_hk_stock(self):
        """测试港股实时价格获取"""
        prices = self.handler.get_realtime_prices(['3690.HK'])
        self.assertIsInstance(prices, dict)
        if '3690.HK' in prices:
            self.assertGreater(prices['3690.HK']['price'], 0)
    
    def test_get_realtime_price_etf(self):
        """测试ETF实时价格获取"""
        prices = self.handler.get_realtime_prices(['159792.SZ'])
        self.assertIsInstance(prices, dict)
        if '159792.SZ' in prices:
            self.assertGreater(prices['159792.SZ']['price'], 0)
    
    def test_fetch_stock_data(self):
        """测试历史数据获取"""
        df = self.handler.fetch_stock_data('300124.SZ', days=30)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)

if __name__ == '__main__':
    unittest.main()
