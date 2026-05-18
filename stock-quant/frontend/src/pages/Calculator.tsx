import { useState, useMemo } from 'react';
import { Card, Form, InputNumber, Button, Descriptions, Row, Col, Tag, Table, Space, Statistic } from 'antd';
import { CalculatorOutlined, DollarOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';

// 主题色
const CARD_BG = '#242830';
const CARD_BORDER = '#3a3f4a';
const GOLD = '#e2b04a';
const TEXT_DIM = 'rgba(255,255,255,0.5)';
const TEXT_LIGHT = 'rgba(255,255,255,0.85)';

const darkCardStyle: React.CSSProperties = {
  background: CARD_BG,
  border: `1px solid ${CARD_BORDER}`,
  borderRadius: 8,
};

interface CalcResult {
  addShares: number;
  addAmount: number;
  newTotalCost: number;
  newCostPrice: number;
  newShares: number;
  currentLoss: number;
  currentLossRate: number;
  newLoss: number;
  newLossRate: number;
  totalInvest: number;
  totalValue: number;
  targetCost: number;
  actualNewCost: number;
  comparisons: ComparisonItem[];
}

interface ComparisonItem {
  label: string;
  addShares: number;
  addAmount: number;
  newCostPrice: number;
  newShares: number;
  newLoss: number;
  newLossRate: number;
  totalInvest: number;
  totalValue: number;
}

function calculateCost(
  costPrice: number,
  shares: number,
  currentPrice: number,
  targetCost: number,
  addPrice: number
): CalcResult | null {
  if (costPrice <= 0 || shares <= 0 || currentPrice <= 0 || targetCost <= 0 || addPrice <= 0) {
    return null;
  }

  const denominator = targetCost - addPrice;
  if (denominator === 0) return null;

  const addSharesRaw = (costPrice * shares - targetCost * shares) / denominator;
  if (addSharesRaw < 0) return null;

  // 补仓数量向上取整到100的倍数
  const addSharesRound = Math.ceil(addSharesRaw / 100) * 100;

  const addAmount = addPrice * addSharesRound;
  const newTotalCost = costPrice * shares + addPrice * addSharesRound;
  const newShares = shares + addSharesRound;
  const newCostPrice = newTotalCost / newShares;

  const currentLoss = (currentPrice - costPrice) * shares;
  const currentLossRate = (currentPrice - costPrice) / costPrice * 100;
  const newLoss = (currentPrice - newCostPrice) * newShares;
  const newLossRate = (currentPrice - newCostPrice) / newCostPrice * 100;

  const totalInvest = costPrice * shares + addPrice * addSharesRound;
  const totalValue = currentPrice * newShares;

  // 方案对比
  const comparisons: ComparisonItem[] = [];
  for (const ratio of [0.5, 1.0, 1.5, 2.0, 3.0]) {
    const compShares = Math.max(100, Math.round(addSharesRound * ratio / 100) * 100);
    const compAmount = addPrice * compShares;
    const compTotalCost = costPrice * shares + addPrice * compShares;
    const compNewShares = shares + compShares;
    const compCostPrice = compTotalCost / compNewShares;
    const compLoss = (currentPrice - compCostPrice) * compNewShares;
    const compLossRate = (currentPrice - compCostPrice) / compCostPrice * 100;
    const compTotalInvest = costPrice * shares + addPrice * compShares;
    const compTotalValue = currentPrice * compNewShares;
    comparisons.push({
      label: `${ratio}x基准补仓`,
      addShares: compShares,
      addAmount: compAmount,
      newCostPrice: compCostPrice,
      newShares: compNewShares,
      newLoss: compLoss,
      newLossRate: compLossRate,
      totalInvest: compTotalInvest,
      totalValue: compTotalValue,
    });
  }

  return {
    addShares: addSharesRound,
    addAmount,
    newTotalCost,
    newCostPrice,
    newShares,
    currentLoss,
    currentLossRate,
    newLoss,
    newLossRate,
    totalInvest,
    totalValue,
    targetCost,
    actualNewCost: newCostPrice,
    comparisons,
  };
}

const compColumns = [
  { title: '方案', dataIndex: 'label', width: 120 },
  { title: '补仓数量', dataIndex: 'addShares', width: 100, render: (v: number) => `${v}股` },
  { title: '补仓金额', dataIndex: 'addAmount', width: 100, render: (v: number) => `¥${v.toFixed(0)}` },
  { title: '新成本价', dataIndex: 'newCostPrice', width: 100, render: (v: number) => <Tag color="blue">¥{v.toFixed(3)}</Tag> },
  { title: '新股数', dataIndex: 'newShares', width: 80, render: (v: number) => `${v}股` },
  { title: '补仓后盈亏', dataIndex: 'newLoss', width: 110, render: (v: number) => <span style={{ color: v >= 0 ? '#3fb950' : '#f85149' }}>{v >= 0 ? '+' : ''}¥{v.toFixed(0)}</span> },
  { title: '补仓后盈亏率', dataIndex: 'newLossRate', width: 110, render: (v: number) => <Tag color={v >= 0 ? 'green' : 'red'}>{v >= 0 ? '+' : ''}{v.toFixed(2)}%</Tag> },
  { title: '总投入', dataIndex: 'totalInvest', width: 100, render: (v: number) => `¥${v.toFixed(0)}` },
  { title: '总市值', dataIndex: 'totalValue', width: 100, render: (v: number) => `¥${v.toFixed(0)}` },
];

export default function Calculator() {
  const [formValues, setFormValues] = useState({
    costPrice: 12.5,
    shares: 1000,
    currentPrice: 10.0,
    targetCost: 11.0,
    addPrice: 10.0,
  });

  const result = useMemo(() => {
    return calculateCost(
      formValues.costPrice,
      formValues.shares,
      formValues.currentPrice,
      formValues.targetCost,
      formValues.addPrice
    );
  }, [formValues.costPrice, formValues.shares, formValues.currentPrice, formValues.targetCost, formValues.addPrice]);

  const handleChange = (field: string, value: number | null) => {
    if (value !== null) {
      setFormValues(prev => ({ ...prev, [field]: value }));
    }
  };

  // 自动将补仓价格设为当前价格
  const syncAddPrice = () => {
    setFormValues(prev => ({ ...prev, addPrice: prev.currentPrice }));
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(180deg, #1e2229 0%, #1a1e25 100%)' }}>
      {/* 导航栏 */}
      <div style={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)', padding: '14px 32px', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center', boxShadow: '0 4px 20px rgba(0,0,0,0.3)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: 1 }}>
            <CalculatorOutlined style={{ marginRight: 10, color: GOLD }} />
            补仓成本计算器
          </h2>
        </div>
        <Space>
          <Link to="/" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none', padding: '6px 14px', borderRadius: 6, fontSize: 13, border: '1px solid rgba(255,255,255,0.15)' }}>
            主页
          </Link>
          <Link to="/forecast7" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none', padding: '6px 14px', borderRadius: 6, fontSize: 13, border: '1px solid rgba(255,255,255,0.15)' }}>
            7天预测
          </Link>
          <Link to="/forecast" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none', padding: '6px 14px', borderRadius: 6, fontSize: 13, border: '1px solid rgba(255,255,255,0.15)' }}>
            预测验证
          </Link>
        </Space>
      </div>

      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 24px 48px' }}>
        <Row gutter={24}>
          {/* 左侧输入 */}
          <Col span={8}>
            <Card style={darkCardStyle} title={<span style={{ color: TEXT_LIGHT }}><CalculatorOutlined /> 输入参数</span>}>
              <Form layout="vertical" size="middle">
                <Form.Item label={<span style={{ color: TEXT_DIM }}>成本价格（每股）</span>}>
                  <InputNumber
                    value={formValues.costPrice}
                    onChange={(v) => handleChange('costPrice', v)}
                    min={0.01}
                    step={0.1}
                    precision={2}
                    prefix="¥"
                    style={{ width: '100%' }}
                  />
                </Form.Item>
                <Form.Item label={<span style={{ color: TEXT_DIM }}>当前持有股数</span>}>
                  <InputNumber
                    value={formValues.shares}
                    onChange={(v) => handleChange('shares', v)}
                    min={100}
                    step={100}
                    style={{ width: '100%' }}
                  />
                </Form.Item>
                <Form.Item label={<span style={{ color: TEXT_DIM }}>当前股票价格</span>}>
                  <InputNumber
                    value={formValues.currentPrice}
                    onChange={(v) => handleChange('currentPrice', v)}
                    min={0.01}
                    step={0.1}
                    precision={2}
                    prefix="¥"
                    style={{ width: '100%' }}
                  />
                </Form.Item>
                <Form.Item label={<span style={{ color: TEXT_DIM }}>目标成本价</span>}>
                  <InputNumber
                    value={formValues.targetCost}
                    onChange={(v) => handleChange('targetCost', v)}
                    min={0.01}
                    step={0.1}
                    precision={2}
                    prefix="¥"
                    style={{ width: '100%' }}
                  />
                </Form.Item>
                <Form.Item label={<span style={{ color: TEXT_DIM }}>补仓价格</span>}>
                  <InputNumber
                    value={formValues.addPrice}
                    onChange={(v) => handleChange('addPrice', v)}
                    min={0.01}
                    step={0.1}
                    precision={2}
                    prefix="¥"
                    style={{ width: '100%' }}
                  />
                  <Button size="small" onClick={syncAddPrice} style={{ marginTop: 4 }}>
                    同步当前价格
                  </Button>
                </Form.Item>
              </Form>
            </Card>
          </Col>

          {/* 右侧结果 */}
          <Col span={16}>
            {result ? (
              <>
                {/* 核心结果 */}
                <Card style={darkCardStyle} title={<span style={{ color: TEXT_LIGHT }}><DollarOutlined /> 补仓方案结果</span>}>
                  <Row gutter={12}>
                    <Col span={6}>
                      <Statistic
                        title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>补仓数量</span>}
                        value={result.addShares}
                        suffix="股"
                        valueStyle={{ color: '#58a6ff', fontSize: 28, fontWeight: 700 }}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>补仓金额</span>}
                        value={result.addAmount}
                        prefix="¥"
                        precision={0}
                        valueStyle={{ color: GOLD, fontSize: 28, fontWeight: 700 }}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>新成本价</span>}
                        value={result.newCostPrice}
                        prefix="¥"
                        precision={3}
                        valueStyle={{ color: '#3fb950', fontSize: 28, fontWeight: 700 }}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>新股数</span>}
                        value={result.newShares}
                        suffix="股"
                        valueStyle={{ color: TEXT_LIGHT, fontSize: 28, fontWeight: 700 }}
                      />
                    </Col>
                  </Row>

                  <div style={{ marginTop: 24 }}>
                    <Descriptions bordered column={2} size="small" labelStyle={{ color: TEXT_DIM, background: '#2a2e36' }} contentStyle={{ color: TEXT_LIGHT, background: CARD_BG }}>
                      <Descriptions.Item label="当前盈亏">
                        <span style={{ color: result.currentLoss >= 0 ? '#3fb950' : '#f85149' }}>
                          {result.currentLoss >= 0 ? '+' : ''}¥{result.currentLoss.toFixed(0)}
                        </span>
                      </Descriptions.Item>
                      <Descriptions.Item label="当前盈亏率">
                        <Tag color={result.currentLossRate >= 0 ? 'green' : 'red'}>
                          {result.currentLossRate >= 0 ? '+' : ''}{result.currentLossRate.toFixed(2)}%
                        </Tag>
                      </Descriptions.Item>
                      <Descriptions.Item label="补仓后盈亏">
                        <span style={{ color: result.newLoss >= 0 ? '#3fb950' : '#f85149' }}>
                          {result.newLoss >= 0 ? '+' : ''}¥{result.newLoss.toFixed(0)}
                        </span>
                      </Descriptions.Item>
                      <Descriptions.Item label="补仓后盈亏率">
                        <Tag color={result.newLossRate >= 0 ? 'green' : 'red'}>
                          {result.newLossRate >= 0 ? '+' : ''}{result.newLossRate.toFixed(2)}%
                        </Tag>
                      </Descriptions.Item>
                      <Descriptions.Item label="总投入">¥{result.totalInvest.toFixed(0)}</Descriptions.Item>
                      <Descriptions.Item label="总市值">¥{result.totalValue.toFixed(0)}</Descriptions.Item>
                      <Descriptions.Item label="目标成本">¥{result.targetCost.toFixed(2)}</Descriptions.Item>
                      <Descriptions.Item label="实际新成本">¥{result.actualNewCost.toFixed(3)}</Descriptions.Item>
                    </Descriptions>
                  </div>

                  {/* 简要提示 */}
                  <div style={{ marginTop: 16, padding: '12px 16px', background: 'rgba(88,166,255,0.08)', borderRadius: 6, border: '1px solid rgba(88,166,255,0.2)' }}>
                    <div style={{ color: TEXT_LIGHT, fontSize: 13 }}>
                      💡 <b>补仓建议：</b>
                      以 ¥{formValues.addPrice.toFixed(2)} 的价格补仓 {result.addShares}股（¥{result.addAmount.toFixed(0)}），
                      可将成本从 ¥{formValues.costPrice.toFixed(2)} 降低到 ¥{result.newCostPrice.toFixed(3)}，
                      补仓后盈亏率从 {result.currentLossRate.toFixed(2)}% 改善为 {result.newLossRate.toFixed(2)}%
                    </div>
                  </div>
                </Card>

                {/* 方案对比 */}
                <Card style={{ ...darkCardStyle, marginTop: 16 }} title={<span style={{ color: TEXT_LIGHT }}>📊 补仓方案对比</span>}>
                  <Table
                    columns={compColumns}
                    dataSource={result.comparisons}
                    rowKey="label"
                    pagination={false}
                    size="small"
                  />
                  <div style={{ color: TEXT_DIM, fontSize: 11, marginTop: 8, textAlign: 'center' }}>
                    基准补仓 = 达到目标成本的最低补仓量（向上取整到100股）| 多倍方案提供更激进降成本选择
                  </div>
                </Card>
              </>
            ) : (
              <Card style={darkCardStyle}>
                <div style={{ textAlign: 'center', padding: 60, color: TEXT_DIM }}>
                  <CalculatorOutlined style={{ fontSize: 48, color: '#555' }} />
                  <p style={{ marginTop: 16 }}>请输入有效参数后自动计算</p>
                  <p style={{ fontSize: 12 }}>注意：目标成本价必须低于当前成本价且高于补仓价格</p>
                </div>
              </Card>
            )}
          </Col>
        </Row>
      </div>
    </div>
  );
}