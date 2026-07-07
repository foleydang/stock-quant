import React, { useMemo } from 'react';
import { Card, InputNumber, Statistic, Row, Col, Tag, Divider, Select } from 'antd';
import { CalculatorOutlined, RiseOutlined, FallOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import axios from 'axios';

const GOLD = '#e2b04a';

interface Position {
  symbol: string;
  name: string;
  shares: number;
  cost: number;
  current: number;
}

export default function Calculator() {
  const [costPrice, setCostPrice] = React.useState<number>(100);
  const [shares, setShares] = React.useState<number>(1000);
  const [currentPrice, setCurrentPrice] = React.useState<number>(80);
  const [targetCost, setTargetCost] = React.useState<number>(90);
  const [positions, setPositions] = React.useState<Position[]>([]);
  const [selected, setSelected] = React.useState<string | undefined>(undefined);

  React.useEffect(() => {
    axios.get('/api/positions')
      .then(res => { if (res.data.status === 'success') setPositions(res.data.positions || []); })
      .catch(() => {});
  }, []);

  // 选中持仓 → 自动填入成本价/持股/现价; 目标成本默认取 现价与成本价中点(便于直接算补仓降本), 均可再改
  const handleSelectPosition = (symbol: string) => {
    const p = positions.find(x => x.symbol === symbol);
    if (!p) return;
    setSelected(symbol);
    setCostPrice(p.cost);
    setShares(p.shares);
    setCurrentPrice(p.current);
    setTargetCost(p.cost > p.current ? +((p.cost + p.current) / 2).toFixed(2) : p.cost);
  };

  // 补仓价格默认等于当前价格
  const addPrice = currentPrice;

  const result = useMemo(() => {
    if (!costPrice || !shares || !currentPrice || !targetCost || costPrice <= 0 || shares <= 0 || currentPrice <= 0) return null;

    const currentLoss = (currentPrice - costPrice) * shares;
    const currentLossRate = (currentPrice - costPrice) / costPrice * 100;
    const totalCost = costPrice * shares;

    // 计算达到目标成本需要的补仓数量
    let addShares: number;
    if (targetCost <= addPrice) {
      addShares = 0; // 目标成本低于当前价格，无法通过当前价格补仓降低成本
    } else if (targetCost <= costPrice) {
      addShares = Math.ceil((totalCost - targetCost * shares) / (targetCost - addPrice) / 100) * 100;
    } else {
      addShares = 0;
    }

    if (addShares < 0) addShares = 0;

    const newShares = shares + addShares;
    const addAmount = addPrice * addShares;
    const newTotalCost = totalCost + addAmount;
    const newCostPrice = newShares > 0 ? newTotalCost / newShares : costPrice;
    const totalValue = currentPrice * newShares;
    const newLoss = totalValue - newTotalCost;
    const newLossRate = newShares > 0 ? (currentPrice - newCostPrice) / newCostPrice * 100 : 0;

    // 补仓方案对比
    const baseAdd = addShares;
    const comparisons = [0.5, 1.0, 1.5, 2.0, 3.0].map(mult => {
      const cShares = Math.ceil(baseAdd * mult / 100) * 100 || Math.ceil(shares * mult * 0.1 / 100) * 100;
      const cAmount = addPrice * cShares;
      const cNewShares = shares + cShares;
      const cNewCost = cNewShares > 0 ? (totalCost + cAmount) / cNewShares : costPrice;
      const cTotalInvest = totalCost + cAmount;
      const cTotalValue = currentPrice * cNewShares;
      const cNewLoss = cTotalValue - cTotalInvest;
      const cNewLossRate = cNewShares > 0 ? (currentPrice - cNewCost) / cNewCost * 100 : 0;
      return {
        label: mult === 1.0 ? '1x 目标补仓' : `${mult}x 补仓`,
        addShares: cShares,
        addAmount: cAmount,
        newCostPrice: cNewCost,
        newLoss: cNewLoss,
        newLossRate: cNewLossRate,
        newShares: cNewShares,
        totalInvest: cTotalInvest,
        totalValue: cTotalValue,
      };
    });

    return {
      costPrice, shares, currentPrice, targetCost, addPrice,
      currentLoss, currentLossRate, totalCost,
      addShares, addAmount, newCostPrice, newShares,
      newTotalCost, totalValue, newLoss, newLossRate,
      comparisons,
    };
  }, [costPrice, shares, currentPrice, targetCost]);

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(180deg, #1e2229 0%, #1a1e25 100%)' }}>
      <div style={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)', padding: '14px 32px', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: 1 }}>
          <CalculatorOutlined style={{ marginRight: 10, color: GOLD }} />
          成本计算器
        </h2>
        <Link to="/" style={{ color: 'rgba(255,255,255,0.8)', textDecoration: 'none', padding: '8px 16px', background: 'rgba(226,176,74,0.15)', borderRadius: 6, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid rgba(226,176,74,0.3)' }}>← 返回主页</Link>
      </div>

      <div style={{ maxWidth: 800, margin: '0 auto', padding: '24px 24px 48px' }}>
        {/* 输入区域 - 四个参数同一行 */}
        <Card style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 20 }} styles={{ body: { padding: 20 } }}>
          <h3 style={{ color: 'rgba(255,255,255,0.85)', marginBottom: 16 }}>输入参数</h3>
          <Row gutter={16} style={{ marginBottom: 16 }}>
            <Col span={12}>
              <div style={{ marginBottom: 6, color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>从我的持仓选择 <span style={{ opacity: 0.7 }}>(自动填入成本价 / 持股 / 现价, 下方仍可修改)</span></div>
              <Select
                value={selected}
                onChange={handleSelectPosition}
                onClear={() => setSelected(undefined)}
                placeholder={positions.length ? '选择持仓股票' : '暂无持仓'}
                style={{ width: '100%' }}
                size="large"
                showSearch
                allowClear
                optionFilterProp="label"
                options={positions.map(p => ({ value: p.symbol, label: `${p.name} (${p.symbol})` }))}
              />
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={6}>
              <div style={{ marginBottom: 6, color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>成本价格</div>
              <InputNumber
                value={costPrice}
                onChange={(v) => setCostPrice(v || 0)}
                prefix="¥"
                style={{ width: '100%' }}
                size="large"
                min={0.01}
                step={0.1}
              />
            </Col>
            <Col span={6}>
              <div style={{ marginBottom: 6, color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>当前持股</div>
              <InputNumber
                value={shares}
                onChange={(v) => setShares(v || 0)}
                suffix="股"
                style={{ width: '100%' }}
                size="large"
                min={100}
                step={100}
              />
            </Col>
            <Col span={6}>
              <div style={{ marginBottom: 6, color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>当前价格</div>
              <InputNumber
                value={currentPrice}
                onChange={(v) => setCurrentPrice(v || 0)}
                prefix="¥"
                style={{ width: '100%' }}
                size="large"
                min={0.01}
                step={0.1}
              />
            </Col>
            <Col span={6}>
              <div style={{ marginBottom: 6, color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>目标成本价</div>
              <InputNumber
                value={targetCost}
                onChange={(v) => setTargetCost(v || 0)}
                prefix="¥"
                style={{ width: '100%' }}
                size="large"
                min={0.01}
                step={0.1}
              />
            </Col>
          </Row>
        </Card>

        {result && (
          <>
            {/* 计算结果 */}
            <Card style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 20 }} styles={{ body: { padding: 20 } }}>
              <h3 style={{ color: 'rgba(255,255,255,0.85)', marginBottom: 16 }}>💡 计算结果</h3>

              {/* 当前状态 */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12, marginBottom: 8 }}>📊 当前持仓状态</div>
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title={<span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>当前盈亏</span>}
                      value={result.currentLoss}
                      prefix="¥"
                      precision={0}
                      valueStyle={{ color: result.currentLoss >= 0 ? '#3fb950' : '#f85149', fontSize: 20 }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title={<span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>盈亏比例</span>}
                      value={result.currentLossRate}
                      suffix="%"
                      precision={2}
                      valueStyle={{ color: result.currentLossRate >= 0 ? '#3fb950' : '#f85149', fontSize: 20 }}
                      prefix={result.currentLossRate >= 0 ? <RiseOutlined /> : <FallOutlined />}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title={<span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>总投入</span>}
                      value={result.totalCost}
                      prefix="¥"
                      precision={0}
                      valueStyle={{ color: 'rgba(255,255,255,0.85)', fontSize: 20 }}
                    />
                  </Col>
                </Row>
              </div>

              <Divider style={{ borderColor: '#3a3f4a' }} />

              {/* 补仓方案 */}
              <div>
                <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12, marginBottom: 8 }}>🎯 补仓方案 (按当前价 ¥{result.addPrice.toFixed(2)} 补仓)</div>
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title={<span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>需要补仓</span>}
                      value={result.addShares}
                      suffix="股"
                      valueStyle={{ color: GOLD, fontSize: 24, fontWeight: 700 }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title={<span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>需要投入</span>}
                      value={result.addAmount}
                      prefix="¥"
                      precision={0}
                      valueStyle={{ color: GOLD, fontSize: 24, fontWeight: 700 }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title={<span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>新成本价</span>}
                      value={result.newCostPrice}
                      prefix="¥"
                      precision={2}
                      valueStyle={{ color: 'rgba(255,255,255,0.85)', fontSize: 24, fontWeight: 700 }}
                    />
                  </Col>
                </Row>

                <Row gutter={16} style={{ marginTop: 12 }}>
                  <Col span={8}>
                    <Statistic
                      title={<span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>补仓后盈亏</span>}
                      value={result.newLoss}
                      prefix="¥"
                      precision={0}
                      valueStyle={{ color: result.newLoss >= 0 ? '#3fb950' : '#f85149', fontSize: 20 }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title={<span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>补仓后盈亏率</span>}
                      value={result.newLossRate}
                      suffix="%"
                      precision={2}
                      valueStyle={{ color: result.newLossRate >= 0 ? '#3fb950' : '#f85149', fontSize: 20 }}
                      prefix={result.newLossRate >= 0 ? <RiseOutlined /> : <FallOutlined />}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title={<span style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>总投入(含补仓)</span>}
                      value={result.newTotalCost}
                      prefix="¥"
                      precision={0}
                      valueStyle={{ color: 'rgba(255,255,255,0.85)', fontSize: 20 }}
                    />
                  </Col>
                </Row>
              </div>

              {result.addShares <= 0 && (
                <div style={{ marginTop: 12, padding: '10px 16px', background: '#1a1a2e', borderRadius: 6 }}>
                  <Tag color={result.targetCost <= result.currentPrice ? 'green' : 'orange'}>
                    {result.targetCost <= result.currentPrice
                      ? `目标成本 ¥${result.targetCost} ≤ 当前价 ¥${result.currentPrice}，当前价补仓无法降低成本`
                      : `目标成本 ¥${result.targetCost} ≥ 原成本 ¥${result.costPrice}，无需补仓降低成本`}
                  </Tag>
                </div>
              )}
            </Card>

            {/* 方案对比 */}
            <Card style={{ background: '#242830', border: '1px solid #3a3f4a' }} styles={{ body: { padding: 20 } }}>
              <h3 style={{ color: 'rgba(255,255,255,0.85)', marginBottom: 16 }}>📊 补仓方案对比</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #3a3f4a' }}>
                      <th style={{ padding: '8px 12px', color: 'rgba(255,255,255,0.5)', textAlign: 'left' }}>方案</th>
                      <th style={{ padding: '8px 12px', color: 'rgba(255,255,255,0.5)', textAlign: 'right' }}>补仓数量</th>
                      <th style={{ padding: '8px 12px', color: 'rgba(255,255,255,0.5)', textAlign: 'right' }}>投入金额</th>
                      <th style={{ padding: '8px 12px', color: 'rgba(255,255,255,0.5)', textAlign: 'right' }}>新成本价</th>
                      <th style={{ padding: '8px 12px', color: 'rgba(255,255,255,0.5)', textAlign: 'right' }}>盈亏率</th>
                      <th style={{ padding: '8px 12px', color: 'rgba(255,255,255,0.5)', textAlign: 'right' }}>总投入</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.comparisons.map((c, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #3a3f4a', background: i === 1 ? 'rgba(226,176,74,0.08)' : 'transparent' }}>
                        <td style={{ padding: '8px 12px', color: i === 1 ? GOLD : 'rgba(255,255,255,0.85)' }}>
                          {c.label} {i === 1 && <Tag color="gold" style={{ marginLeft: 6 }}>推荐</Tag>}
                        </td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', color: 'rgba(255,255,255,0.85)' }}>{c.addShares}股</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', color: 'rgba(255,255,255,0.85)' }}>¥{c.addAmount.toFixed(0)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', color: i === 1 ? GOLD : 'rgba(255,255,255,0.85)', fontWeight: i === 1 ? 700 : 400 }}>¥{c.newCostPrice.toFixed(2)}</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', color: c.newLossRate >= 0 ? '#3fb950' : '#f85149' }}>{c.newLossRate.toFixed(2)}%</td>
                        <td style={{ padding: '8px 12px', textAlign: 'right', color: 'rgba(255,255,255,0.85)' }}>¥{c.totalInvest.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}