# 部署: hkserver 全包 + Mac sqlite3_rsync 副本

## 架构

```
              抓数(Tushare日线/新浪30m/yfinance港股ETF/东财) + 预测 + 监控 + API
                                    │
                            ┌───────▼────────┐
                            │   hkserver     │  ← 唯一真身库 stock_data.db
                            │ (47.242.x, HK) │
                            └───────▲────────┘
                    sqlite3_rsync 拉│ │推 模型产物 (rsync)
                       (只传变化页) │ │ (几十MB)
                            ┌───────┴─┴──────┐
                            │      Mac       │  ← 训练 (本地全表扫描)
                            └────────────────┘
   hzserver 退役 ·  OSS 分片(oss_incr.py)下线, 保留为可选灾备
```

数据只有一个流向:**Mac 主动拉**(Mac 在 NAT 后无公网 IP,hkserver 推不过来)。

---

## Step 0 — 连通性闸门(必须先过!)

hkserver 是境外机,先确认它能抓到境内数据源。**任一失败就先别切,回来找我调整**(30m 可能要留境内):

```bash
ssh hkserver "curl -s -m10 -o /dev/null -w 'sina:%{http_code} %{time_total}s\n' 'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol=sh600000&scale=30&datalen=5'; curl -s -m10 -o /dev/null -w 'eastmoney:%{http_code}\n' 'https://push2.eastmoney.com/api/qt/stock/get?secid=1.600000&fields=f43'; curl -s -m10 -o /dev/null -w 'tushare:%{http_code}\n' https://api.tushare.pro"
```

期望:三个都 `200`,且 `sina` 耗时 <1s。

---

## Step 1 — 安装 sqlite3_rsync (Mac + hkserver 两台都要)

不在包管理里,从 SQLite 源码编个单文件工具(3.53.3,版本以 sqlite.org 最新为准):

```bash
cd /tmp
curl -O https://sqlite.org/2026/sqlite-src-3530300.zip
unzip -q sqlite-src-3530300.zip && cd sqlite-src-3530300
./configure && make sqlite3_rsync
# Mac (Apple Silicon, 免 sudo): cp sqlite3_rsync /opt/homebrew/bin/
# hkserver (root): cp sqlite3_rsync /usr/local/bin/
which sqlite3_rsync                              # 验证在 PATH 上
```
> hkserver 已确认有 gcc+make,可直接编译。

> 嫌编译麻烦的免编译替代:Litestream(单二进制,hkserver `replicate` 到 OSS,Mac `restore`)。要走这条找我改脚本。

## Step 2 — Mac → hkserver 免密 SSH

`~/.ssh/config` 已有 `hkserver` 别名。确保公钥已在 hkserver:

```bash
ssh-copy-id hkserver            # 若还没配
ssh hkserver 'echo ok'          # 应免密返回 ok
```

## Step 3 — 首次把真身库落到 hkserver

从当前最全的机器(Mac 或退役前的 hzserver)一次性把库放上去:

```bash
# 以 Mac 现有库为准 (1.4G, 首次全量, 之后都是增量)
sqlite3_rsync ~/github/stock-quant/python/data/stock_data.db \
              hkserver:/root/github/stock-quant/python/data/stock_data.db
```

hkserver 上确认 `.env`(含 `TUSHARE_TOKEN` 等)、conda python 就位,代码 `git pull` 到位。

## Step 4 — hkserver 配 cron

```cron
# 盘中每 30 分钟抓 30m (脚本内部再判交易时段)
*/30 9-15 * * 1-5  /root/github/stock-quant/scripts/hkserver_intraday.sh >> /root/logs/intraday.log 2>&1
# 盘后一次: 日线 + 港股ETF + 预测
30  15  * * 1-5    /root/github/stock-quant/scripts/hkserver_eod.sh      >> /root/logs/eod.log 2>&1
```

监控 / API 服务也在 hkserver 起(本地读真身库最快)。

## Step 5 — Mac 训练(按需或定时)

```bash
bash scripts/mac_pull_and_train.sh            # 拉最新库 -> 训练 -> 回推模型
bash scripts/mac_pull_and_train.sh --quick    # 透传给 retrain_all_mac.py
```

## Step 6 — 退役 hzserver / 下线 OSS

- 删/停 hzserver 上的 `sync_and_predict.sh` 等 cron。
- 停 Mac 上的 `mac_upload_loop.sh`(不再往 OSS 传 30m)。
- `oss_incr.py` / `sync_db.sh` / `upload_to_oss.sh` 不再进主链路,**保留作可选灾备**(想额外备份到 OSS 时手动 `oss_incr.py upload --producer server`)。

---

## 日常

- 每天 hkserver 自动抓数+预测,库始终最新。
- 训练时 Mac 跑 `mac_pull_and_train.sh`:秒级增量拉库 → 训练 → 模型自动回推 hkserver 生效。
- 换新机做训练:装 sqlite3_rsync + 配免密,首次 `sqlite3_rsync hkserver:...db localdb`(全量),之后增量。
