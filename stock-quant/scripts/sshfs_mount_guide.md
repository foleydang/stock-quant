# SSHFS 方案 - Mac 直接操作服务器 SQLite

## Mac 安装 SSHFS

```bash
brew install sshfs
```

## 挂载服务器目录

```bash
# 创建挂载点
mkdir -p ~/mnt/stock-server

# 挂载
sshfs root@YOUR_SERVER_IP:/root/github/stock-quant/stock-quant/python/data ~/mnt/stock-server

# 用完卸载
umount ~/mnt/stock-server  # 或 fusermount -u ~/mnt/stock-server
```

## Mac 本地操作远程数据库

挂载后，SQLite 文件就在本地路径：

```python
# Mac 上运行的代码
import sqlite3

# 直接连接"本地"文件（实际是远程服务器）
conn = sqlite3.connect('~/mnt/stock-server/stock_data.db')

# 写入数据（直接写入远程服务器）
cursor = conn.cursor()
cursor.execute("INSERT INTO kline_30m VALUES (...)")
conn.commit()
```

## 定时更新脚本（Mac）

```bash
#!/bin/bash
# ~/github/stock-quant/scripts/mac_update_via_sshfs.sh

# 1. 挂载（如果未挂载）
mountpoint -q ~/mnt/stock-server || sshfs root@SERVER_IP:/root/github/stock-quant/stock-quant/python/data ~/mnt/stock-server

# 2. 运行 Python 更新数据
cd ~/github/stock-quant/stock-quant/python
python3 update_via_mount.py --db ~/mnt/stock-server/stock_data.db

# 3. 可选：卸载
# umount ~/mnt/stock-server

echo "数据更新完成: $(date)"
```

## 优势

- ✅ Mac 直接操作远程 SQLite
- ✅ 无需额外服务（只需 SSH）
- ✅ 免费
- ✅ 实时写入

## 注意

- ⚠️ 挂载后 SQLite 操作会有网络延迟
- ⚠️ 不要同时操作（服务器和 Mac 同时写入可能冲突）
- ⚠️ 建议只在 Mac 更新时操作，服务器只读
