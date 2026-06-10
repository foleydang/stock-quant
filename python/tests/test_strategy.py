"""
策略测试
"""
import unittest
import sys
sys.path.insert(0, '..')
import pandas as pd

from strategy.train_lgb_enhanced import EnhancedFeatureEngineer

class TestStrategy(unittest.TestCase):
    """策略测试"""
    
    def test_calculate_features(self):
        """测试特征计算"""
        # 创建模拟数据
        data = {
            'date': pd.date_range('2025-01-01', periods=250, freq='30min'),
            'open': [100 + i*0.1 for i in range(250)],
            'high': [100.5 + i*0.1 for i in range(250)],
            'low': [99.5 + i*0.1 for i in range(250)],
            'close': [100 + i*0.1 for i in range(250)],
            'volume': [1000000 for i in range(250)]
        }
        df = pd.DataFrame(data)
        
        features = EnhancedFeatureEngineer.calculate_features(df)
        self.assertGreater(len(features), 0)
        self.assertGreater(len(features.columns), 20)  # 应有20+特征

if __name__ == '__main__':
    unittest.main()
