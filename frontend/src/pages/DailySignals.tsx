import { useState, useEffect } from 'react';
import { Card, Table, Tag, Spin, Statistic, Row, Col, TableColumnsType } from 'antd';
import { RiseOutlined, FallOutlined, ThunderboltOutlined, AimOutlined } from '@ant-design/icons';
import axios from 'axios';

const CARD_BG = '#242830';
const CARD_BORDER = '#3a3f4a';
const GOLD = '#e2b04a';
const TEXT_DIM = 'rgba(255,255,255,0.5)';

interface StockSignal {
  rank: number;
  symbol: string;
  name: string;
  score: number;
  signal: string;
  upProb?: number;
  tpProb?: number;
  candidate?: boolean;
}

interface DailySignals {
  predDate: string;
  totalStocks: number;
  caveat?: string;
  generatedAt?: string;
  cached?: boolean;
  distribution: {
    strong_buy: number;
    buy: number;
    hold: number;
    sell: number;
    strong_sell: number;
  };
  signals: {
    strong_buy: StockSignal[];
    buy: StockSignal[];
    sell: StockSignal[];
    strong_sell: StockSignal[];
  };
}

export default function DailySignals() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<DailySignals | null>(null);

  useEffect(() => { fetchData(); }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await axios.get('/api/advisor/scan');
      if (res.data.status === 'success') setData(res.data);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const signalColumns: TableColumnsType<StockSignal> = [
    { title: '排名', dataIndex: 'rank', width: 60, render: (r: number) => <span style={{ color: GOLD }}>#{r}</span> },
    { title: '代码', dataIndex: 'symbol', width: 100 },
    { title: '名称', dataIndex: 'name', width: 100 },
    { title: '预测20日收益', dataIndex: 'score', width: 110, render: (s: number) => <span style={{ color: s > 0 ? '#52c41a' : '#ff4d4f', fontWeight: 'bold' }}>{s > 0 ? '+' : ''}{(s * 100).toFixed(2)}%</span> },
    { title: '上涨概率', dataIndex: 'upProb', width: 90, render: (p?: number) => p == null ? '-' : <span style={{ color: TEXT_DIM }}>{(p * 100).toFixed(1)}%</span> },
    { title: '超卖候选', dataIndex: 'candidate', width: 90, render: (c?: boolean) => c ? <Tag color="gold">补仓候选</Tag> : <span style={{ color: TEXT_DIM }}>-</span> },
    { title: '信号', dataIndex: 'signal', width: 100, render: (s: string) => {
      if (s.includes('强烈买入')) return <Tag color="green"><ThunderboltOutlined /> 强烈买入</Tag>;
      if (s.includes('买入')) return <Tag color="cyan">买入</Tag>;
      if (s.includes('强烈卖出')) return <Tag color="red"><ThunderboltOutlined /> 强烈卖出</Tag>;
      if (s.includes('卖出')) return <Tag color="orange">卖出</Tag>;
      return <Tag>{s}</Tag>;
    }},
  ];

  if (loading) return <Spin tip="加载预测信号..." style={{ display: 'block', margin: '60px auto' }} />;
  if (!data) return <Card style={{ textAlign: 'center', padding: 40, background: CARD_BG, border: `1px solid ${CARD_BORDER}` }}><p style={{ color: TEXT_DIM }}>暂无预测数据，请先运行预测模型</p></Card>;

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(180deg, #1e2229 0%, #1a1e25 100%)' }}>
      <div style={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)', padding: '14px 32px', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: 1 }}>
          <AimOutlined style={{ marginRight: 10, color: GOLD }} /> 每日预测信号
        </h2>
        <span style={{ color: TEXT_DIM, fontSize: 13 }}>数据日期: {data.predDate ? `${data.predDate.slice(0,4)}-${data.predDate.slice(4,6)}-${data.predDate.slice(6,8)}` : '—'} | 共 {data.totalStocks} 只股票{data.cached ? ' | 缓存' : ''}</span>
      </div>

      <div style={{ maxWidth: 1400, margin: '0 auto', padding: 24 }}>
        {data.caveat && (
          <div style={{ background: 'rgba(226,176,74,0.08)', border: `1px solid ${GOLD}`, borderRadius: 6, padding: '10px 14px', marginBottom: 16, color: GOLD, fontSize: 13 }}>
            ⚠️ {data.caveat}
          </div>
        )}
        {/* 概览 */}
        <Card style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}`, marginBottom: 16 }} styles={{ body: { padding: 14 } }}>
          <Row gutter={16}>
            <Col span={4}>
              <Statistic title={<span style={{ color: TEXT_DIM }}>🟢 强烈买入</span>} value={data.distribution.strong_buy} suffix="只" valueStyle={{ color: '#52c41a', fontSize: 28 }} prefix={<ThunderboltOutlined />} />
            </Col>
            <Col span={4}>
              <Statistic title={<span style={{ color: TEXT_DIM }}>🟢 买入</span>} value={data.distribution.buy} suffix="只" valueStyle={{ color: '#73d13d', fontSize: 28 }} prefix={<RiseOutlined />} />
            </Col>
            <Col span={4}>
              <Statistic title={<span style={{ color: TEXT_DIM }}>🟡 持有</span>} value={data.distribution.hold} suffix="只" valueStyle={{ color: '#faad14', fontSize: 28 }} />
            </Col>
            <Col span={4}>
              <Statistic title={<span style={{ color: TEXT_DIM }}>🔴 卖出</span>} value={data.distribution.sell} suffix="只" valueStyle={{ color: '#ff7a45', fontSize: 28 }} prefix={<FallOutlined />} />
            </Col>
            <Col span={4}>
              <Statistic title={<span style={{ color: TEXT_DIM }}>🔴 强烈卖出</span>} value={data.distribution.strong_sell} suffix="只" valueStyle={{ color: '#ff4d4f', fontSize: 28 }} prefix={<ThunderboltOutlined />} />
            </Col>
          </Row>
        </Card>

        {/* 买入信号 */}
        <Card title={<span style={{ color: '#52c41a' }}><RiseOutlined /> 买入推荐 (Top 15)</span>} style={{ background: CARD_BG, border: '1px solid #3a3f4a', marginBottom: 16 }}>
          <Table columns={signalColumns} dataSource={[...data.signals.strong_buy, ...data.signals.buy].slice(0, 15)} rowKey="symbol" pagination={false} size="small" />
        </Card>

        {/* 卖出信号 */}
        <Card title={<span style={{ color: '#ff4d4f' }}><FallOutlined /> 卖出/回避 (Top 15)</span>} style={{ background: CARD_BG, border: '1px solid #3a3f4a' }}>
          <Table columns={signalColumns} dataSource={[...data.signals.strong_sell, ...data.signals.sell].slice(0, 15)} rowKey="symbol" pagination={false} size="small" />
        </Card>
      </div>
    </div>
  );
}