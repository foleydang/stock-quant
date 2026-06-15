#!/usr/bin/env python3
"""
LSTM 时序特征提取器 v1

原理:
  LGBM 看到的是"今天的特征 → 预测未来排名"，每个交易日独立
  LSTM 捕捉的是"过去60天的价格走势形态 → 未来方向"
  两者正交，LSTM 输出作为额外特征拼入 LGBM

架构:
  输入: 过去60天 × 15个基础特征 (OHLCV衍生)
  编码: 2层 LSTM (hidden=64) → 64维时序embedding
  训练目标: 预测未来5日收益率 (回归)
  输出: 64维特征向量，拼入 LGBM 特征矩阵

硬件:
  M4 Pro 18GB, MPS加速 → 训练30min, 推理10min
  仅需 CPU+GPU, 不用额外设备
"""

import os, sys, sqlite3, pickle, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

# 设备选择
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("🚀 使用 MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    print("⚠️ 使用 CPU (慢)")

# 路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data/stock_data.db')
MODEL_DIR = os.path.join(ROOT, 'models/lstm_encoder')
os.makedirs(MODEL_DIR, exist_ok=True)

# ============ 配置 ============
SEQ_LEN = 60        # 回溯60个交易日
HORIZON = 5         # 预测未来5日
HIDDEN_DIM = 64     # LSTM隐藏维度 → 输出64维特征
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 512
EPOCHS = 50
LR = 0.001
WEIGHT_DECAY = 1e-4

# 时序切分 (与 LGBM 训练一致)
TRAIN_CUTOFF = '2023-12-18'
VAL_CUTOFF = '2025-03-24'

# LSTM 输入特征 (从 OHLCV 衍生, 共15个)
def compute_lstm_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """从日线 OHLCV 计算 LSTM 输入特征 (15维)"""
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    open_ = df['open'].astype(float)
    volume = df['volume'].astype(float)

    f = pd.DataFrame(index=df.index)
    # 价格收益率
    f['ret_1'] = close.pct_change(1)
    f['ret_5'] = close.pct_change(5)
    f['ret_20'] = close.pct_change(20)
    # 对数收益
    f['logret_1'] = np.log(close / close.shift(1))
    # 波动率
    returns = close.pct_change()
    f['vol_5'] = returns.rolling(5).std()
    f['vol_20'] = returns.rolling(20).std()
    # 成交量
    f['vol_ratio'] = volume / (volume.rolling(20).mean() + 1e-10)
    f['vol_chg'] = volume.pct_change(5)
    # 价格形态
    f['hl_ratio'] = (high - low) / (close + 1e-10)
    f['co_ratio'] = close / (open_ + 1e-10) - 1
    # 技术指标
    f['rsi_14'] = compute_rsi(close, 14)
    f['pos_20'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-10)
    # 均线偏离
    f['ma20_dev'] = close / close.rolling(20).mean() - 1
    f['ma60_dev'] = close / close.rolling(60).mean() - 1

    return f

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-10))


# ============ 数据集 ============
class SeqDataset(Dataset):
    def __init__(self, sequences, targets):
        self.seq = torch.FloatTensor(sequences)
        self.tgt = torch.FloatTensor(targets).unsqueeze(1)
    def __len__(self): return len(self.seq)
    def __getitem__(self, i): return self.seq[i], self.tgt[i]


# ============ 模型 ============
class LSTMPredictor(nn.Module):
    """LSTM 回归预测器: 预测未来收益率, 同时输出embedding"""
    def __init__(self, input_dim=15, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """x: (batch, seq_len, input_dim)"""
        out, (h, _) = self.lstm(x)
        self.embedding = h[-1]  # 最后一层隐状态 = 64维embedding
        return self.fc(self.embedding)

    def encode(self, x):
        """仅提取embedding (推理用)"""
        with torch.no_grad():
            out, (h, _) = self.lstm(x)
        return h[-1].cpu().numpy()


# ============ 数据准备 ============
def load_and_prepare() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """加载数据, 构建序列, 返回 (train_X, train_y, val_X, val_y, feature_names)"""
    print("📊 加载日线数据...")
    conn = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_daily")]
    print(f"   {len(symbols)} 只股票")

    all_seqs, all_targets = [], []
    train_seqs, train_targets = [], []
    val_seqs, val_targets = [], []

    n_ok, n_skip, n_err = 0, 0, 0
    for sym in symbols:
        try:
            df = pd.read_sql(
                "SELECT date, open, high, low, close, volume FROM kline_daily "
                "WHERE symbol=? ORDER BY date", conn, params=(sym,))
            if len(df) < SEQ_LEN + HORIZON + 20:
                n_skip += 1
                continue
            df['date'] = pd.to_datetime(df['date'], format='mixed')

            # 计算 LSTM 输入特征
            feats = compute_lstm_inputs(df)
            feats = feats.fillna(0)
            feats = feats.replace([np.inf, -np.inf], 0)
            feat_arr = feats.values.astype(np.float32)

            # 计算目标: 未来5日收益率
            close = df['close'].astype(float).values
            target = np.zeros(len(close))
            for i in range(len(close) - HORIZON):
                target[i] = (close[i + HORIZON] - close[i]) / (close[i] + 1e-10)

            # 构建滑动窗口
            for i in range(SEQ_LEN, len(feat_arr) - HORIZON):
                seq = feat_arr[i - SEQ_LEN:i]
                tgt = target[i]

                # 过滤异常值
                if abs(tgt) > 0.20 or np.isnan(tgt) or np.isnan(seq).any():
                    continue

                date = df['date'].iloc[i]
                date_str = pd.Timestamp(date).strftime('%Y-%m-%d')

                if date_str <= TRAIN_CUTOFF:
                    train_seqs.append(seq)
                    train_targets.append(tgt)
                elif date_str <= VAL_CUTOFF:
                    val_seqs.append(seq)
                    val_targets.append(tgt)
            n_ok += 1
        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print(f"   ⚠️ {sym}: {e}")

    print(f"   成功: {n_ok} | 数据不足: {n_skip} | 错误: {n_err}")

    conn.close()

    train_X = np.array(train_seqs, dtype=np.float32)
    train_y = np.array(train_targets, dtype=np.float32)
    val_X = np.array(val_seqs, dtype=np.float32)
    val_y = np.array(val_targets, dtype=np.float32)

    print(f"   训练: {len(train_X):,} 条 | 验证: {len(val_X):,} 条")
    # LSTM 输入列名 (与 compute_lstm_inputs 保持一致)
    lstm_feature_names = ['ret_1','ret_5','ret_20','logret_1','vol_5','vol_20',
                          'vol_ratio','vol_chg','hl_ratio','co_ratio',
                          'rsi_14','pos_20','ma20_dev','ma60_dev']
    return train_X, train_y, val_X, val_y, lstm_feature_names


# ============ 训练 ============
def train_model(model, train_loader, val_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    patience = 0
    best_path = os.path.join(MODEL_DIR, 'best_model.pt')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if torch.isnan(loss):
                print(f"   ❌ NaN loss @ epoch {epoch}, 数据可能有问题")
                return model, float('nan')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(x), y).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d} | train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= 10:
                print(f"  早停 @ epoch {epoch}")
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
    return model, best_loss


# ============ 推理: 提取 embedding ============
def extract_embeddings(model, output_path: str, norm_stats=None):
    """为所有股票的所有日期提取 LSTM embedding, 存入 .pkl"""
    print("\n🔮 提取 LSTM embeddings...")
    model.eval()
    model.to(DEVICE)

    conn = sqlite3.connect(DB_PATH)
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM kline_daily")]
    print(f"   {len(symbols)} 只股票")

    # 加载标准化参数
    if norm_stats is None:
        stats_path = os.path.join(MODEL_DIR, 'norm_stats.pt')
        if os.path.exists(stats_path):
            norm_stats = torch.load(stats_path, map_location='cpu', weights_only=False)

    embeddings = {}  # {symbol: {date_str: np.array(64,)}}

    for sym in symbols:
        try:
            df = pd.read_sql(
                "SELECT date, open, high, low, close, volume FROM kline_daily "
                "WHERE symbol=? ORDER BY date", conn, params=(sym,))
            if len(df) < SEQ_LEN + 20:
                continue
            df['date'] = pd.to_datetime(df['date'], format='mixed')

            feats = compute_lstm_inputs(df).fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

            # 应用标准化 (squeeze去掉 keepdims 维度, 适配 2D 推理数据)
            if norm_stats is not None:
                mean = np.squeeze(norm_stats['x_mean'])  # (1,1,14) → (14,)
                std = np.squeeze(norm_stats['x_std'])
                feats = np.nan_to_num((feats - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

            emb_dict = {}
            for i in range(SEQ_LEN, len(feats)):
                seq = feats[i - SEQ_LEN:i]
                if np.isnan(seq).any():
                    continue
                x = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
                emb = model.encode(x).flatten()
                date_str = str(df['date'].iloc[i])[:10]
                emb_dict[date_str] = emb.astype(np.float32)

            embeddings[sym] = emb_dict
        except Exception:
            continue

        if len(embeddings) % 100 == 0:
            print(f"   {len(embeddings)}/{len(symbols)}")

    conn.close()

    # 保存
    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    total = sum(len(v) for v in embeddings.values())
    print(f"\n   ✅ 已保存: {output_path}")
    print(f"   {len(embeddings)} 只股票, {total:,} 个 embedding")
    print(f"   文件大小: {os.path.getsize(output_path)/1024/1024:.1f} MB")


# ============ 主入口 ============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract-only', type=str, default=None,
                       help='跳过训练, 从已有模型提取embedding (指定模型路径)')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--hidden', type=int, default=HIDDEN_DIM)
    args = parser.parse_args()

    if args.extract_only:
        # 仅推理模式
        model = LSTMPredictor(hidden_dim=args.hidden)
        model.load_state_dict(torch.load(args.extract_only, map_location=DEVICE, weights_only=True))
        out_path = os.path.join(ROOT, 'data/lstm_embeddings.pkl')
        extract_embeddings(model, out_path)
        return

    print("=" * 60)
    print(f"  LSTM 时序编码器训练")
    print(f"  回溯: {SEQ_LEN}天 | 预测: {HORIZON}天 | 隐藏: {args.hidden}")
    print(f"  设备: {DEVICE}")
    print("=" * 60)

    # 1. 准备数据
    t0 = time.time()
    train_X, train_y, val_X, val_y, feature_names = load_and_prepare()
    print(f"  数据准备耗时: {time.time()-t0:.0f}s")

    # 1.5 数据标准化 (防止 NaN loss)
    print("  标准化数据...")
    # 按特征维度计算 mean/std, 处理全零特征
    train_mean = np.nanmean(train_X, axis=(0, 1), keepdims=True)
    train_std = np.nanstd(train_X, axis=(0, 1), keepdims=True)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)  # 零方差特征用1
    train_X = np.nan_to_num((train_X - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0)
    val_X = np.nan_to_num((val_X - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0)
    # 目标也标准化
    y_mean = np.nanmean(train_y)
    y_std = np.nanstd(train_y) + 1e-8
    train_y = np.nan_to_num((train_y - y_mean) / y_std, nan=0.0)
    val_y = np.nan_to_num((val_y - y_mean) / y_std, nan=0.0)
    print(f"   训练集均值: {train_X.mean():.4f} std: {train_X.std():.4f}")
    print(f"   目标均值: {y_mean:.4f} std: {y_std:.4f}")

    train_ds = SeqDataset(train_X, train_y)
    val_ds = SeqDataset(val_X, val_y)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False)

    # 2. 训练
    model = LSTMPredictor(
        input_dim=train_X.shape[2],
        hidden_dim=args.hidden,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    print(f"\n  模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n🏋️ 训练 LSTM...")
    t0 = time.time()
    model, best_loss = train_model(model, train_loader, val_loader, args.epochs)
    print(f"  训练耗时: {time.time()-t0:.0f}s | best val loss: {best_loss:.6f}")

    # 3. 保存模型 + 标准化参数
    model_path = os.path.join(MODEL_DIR, 'encoder.pt')
    torch.save(model.state_dict(), model_path)
    # 保存标准化参数 (推理时复用)
    stats = {'x_mean': train_mean, 'x_std': train_std, 'y_mean': y_mean, 'y_std': y_std}
    torch.save(stats, os.path.join(MODEL_DIR, 'norm_stats.pt'))
    print(f"\n💾 模型已保存: {model_path}")

    # 4. 提取 embedding
    emb_path = os.path.join(ROOT, 'data/lstm_embeddings.pkl')
    extract_embeddings(model, emb_path)

    print("\n✅ 完成! 下一步: python strategy/train.py --model daily")


if __name__ == '__main__':
    main()