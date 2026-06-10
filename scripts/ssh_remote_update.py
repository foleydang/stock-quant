"""
Mac 通过 SSH 远程在服务器上更新数据

原理：
1. Mac 本地拉取东方财富数据 → 保存为 JSON/CSV
2. SSH 上传到服务器临时目录
3. SSH 执行服务器上的 Python 脚本导入数据库
"""

import subprocess
import json
import os
import tempfile
import paramiko  # pip install paramiko

class SSHDatabaseUpdater:
    """SSH 远程更新数据库"""
    
    def __init__(self, host, username='root', key_file=None):
        self.host = host
        self.username = username
        self.key_file = key_file or '~/.ssh/id_rsa'
        
    def upload_and_import(self, data: dict, symbol: str):
        """
        上传数据并导入数据库
        
        Args:
            data: K线数据 dict
            symbol: 股票代码
        """
        # 1. 本地保存临时文件
        tmp_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        json.dump(data, tmp_file)
        tmp_file.close()
        
        # 2. SCP 上传
        remote_tmp = f'/tmp/{symbol}_update.json'
        subprocess.run([
            'scp', '-i', self.key_file,
            tmp_file.name,
            f'{self.username}@{self.host}:{remote_tmp}'
        ], check=True)
        
        # 3. SSH 执行导入脚本
        import_cmd = f'''
cd /root/github/stock-quant/python
python3.8 << 'PYEOF'
import json
import sqlite3
import pandas as pd

data = json.load(open('{remote_tmp}'))
conn = sqlite3.connect('data/stock_data.db')
cursor = conn.cursor()

for row in data['klines']:
    cursor.execute("""
        INSERT OR REPLACE INTO kline_30m 
        (symbol, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, ('{symbol}', row['date'], row['open'], row['high'], 
          row['low'], row['close'], row['volume']))
conn.commit()
print("导入完成:", len(data['klines']), "条")
PYEOF
rm {remote_tmp}
'''
        
        subprocess.run([
            'ssh', '-i', self.key_file,
            f'{self.username}@{self.host}',
            import_cmd
        ], check=True)
        
        # 4. 清理本地临时文件
        os.unlink(tmp_file.name)
        
        print(f'{symbol}: ✅ 已上传并导入数据库')


# 使用示例
if __name__ == '__main__':
    updater = SSHDatabaseUpdater(
        host='YOUR_SERVER_IP',
        key_file='~/.ssh/id_rsa'
    )
    
    # 假设你已经从东方财富获取了数据
    test_data = {
        'klines': [
            {'date': '2026-04-28 09:30', 'open': 39.5, 'high': 39.6, 'low': 39.4, 'close': 39.5, 'volume': 100000},
        ]
    }
    
    updater.upload_and_import(test_data, '600036.SH')
