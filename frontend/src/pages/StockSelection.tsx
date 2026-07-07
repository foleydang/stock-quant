import { useState } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend,
} from 'chart.js';
import { Button, Card, Row, Col, Tag, Table, Spin, message } from 'antd';
import { StockOutlined, ReloadOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import axios from 'axios';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const GOLD = '#e2b04a';

interface ScanItem {
  rank: number;
  symbol: string;
  name: string;
  score: number;   // 预测 N 日收益 (小数, 如 0.052)
  signal: string;
  upProb: number;  // 0~1
  tpProb: number;  // 0~1
  candidate: boolean;
}
interface ScanPayload {
  status: string;
  predDate?: string;
  totalStocks?: number;
  distribution?: Record<string, number>;
  signals?: { strong_buy?: ScanItem[]; buy?: ScanItem[]; sell?: ScanItem[]; strong_sell?: ScanItem[] };
  horizon?: number;
  caveat?: string;
  generatedAt?: string;
  cached?: boolean;
  cacheAgeMin?: number;
}

const signalTag = (v: string) => {
  if (v.includes('强烈买')) return <Tag color="green">{v}</Tag>;
  if (v.includes('买')) return <Tag color="lime">{v}</Tag>;
  if (v.includes('强烈卖')) return <Tag color="red">{v}</Tag>;
  if (v.includes('卖')) return <Tag color="orange">{v}</Tag>;
  return <Tag>{v}</Tag>;
};

export default function StockSelection() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ScanPayload | null>(null);

  const runScan = async () => {
    setLoading(true);
    try {
      // 手动触发: refresh=1 强制用 add_advisor 重算全 A 股截面排名 (可能耗时数十秒)
      const res = await axios.get('/api/advisor/scan?refresh=1');
      if (res.data.status === 'success') {
        setData(res.data);
        message.success(`扫描完成, 共 ${res.data.totalStocks ?? 0} 只 A 股参与截面排名`);
      } else {
        setData(null);
        message.error(res.data.message || '扫描失败');
      }
    } catch {
      setData(null);
      message.error('无法连接到服务器');
    } finally {
      setLoading(false);
    }
  };

  const horizon = data?.horizon ?? 20;
  const buys: ScanItem[] = [...(data?.signals?.strong_buy || []), ...(data?.signals?.buy || [])];
  const sells: ScanItem[] = [...(data?.signals?.sell || []), ...(data?.signals?.strong_sell || [])];
  const topBuys = buys.slice(0, 15);

  const columns = [
    { title: '排名', dataIndex: 'rank', width: 60 },
    { title: '股票', dataIndex: 'name', width: 120 },
    { title: '代码', dataIndex: 'symbol', width: 110 },
    { title: '信号', dataIndex: 'signal', width: 100, render: (v: string) => signalTag(v) },
    { title: `预测${horizon}日收益`, dataIndex: 'score', width: 120, render: (r: number) => <span style={{ color: r >= 0 ? '#3fb950' : '#f85149', fontWeight: 700 }}>{r >= 0 ? '+' : ''}{(r * 100).toFixed(2)}%</span> },
    { title: '上涨概率', dataIndex: 'upProb', width: 90, render: (p: number) => <span style={{ color: p >= 0.5 ? '#3fb950' : 'rgba(255,255,255,0.7)' }}>{(p * 100).toFixed(1)}%</span> },
    { title: '止盈概率', dataIndex: 'tpProb', width: 90, render: (p: number) => <span style={{ color: 'rgba(255,255,255,0.7)' }}>{(p * 100).toFixed(1)}%</span> },
    { title: '补仓候选态', dataIndex: 'candidate', width: 110, render: (c: boolean) => c ? <Tag color="gold">候选</Tag> : <Tag>—</Tag> },
  ];

  const darkLegend = { labels: { color: 'rgba(255,255,255,0.7)' } };
  const darkTicks = { color: 'rgba(255,255,255,0.5)' };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(180deg, #1e2229 0%, #1a1e25 100%)' }}>
      <div style={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)', padding: '14px 32px', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: 1 }}>
          <StockOutlined style={{ marginRight: 10, color: GOLD }} /> 智能选股 · 截面排名
        </h2>
        <Link to="/" style={{ color: 'rgba(255,255,255,0.8)', textDecoration: 'none', padding: '8px 16px', background: 'rgba(226,176,74,0.15)', borderRadius: 6, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid rgba(226,176,74,0.3)' }}>← 返回主页</Link>
      </div>

      <div style={{ maxWidth: 1200, margin: '0 auto', padding: 24 }}>
        {/* 诚实 caveat */}
        <div style={{ background: 'rgba(226,176,74,0.08)', border: `1px solid ${GOLD}`, borderRadius: 6, padding: '10px 14px', marginBottom: 16, color: GOLD, fontSize: 13, lineHeight: 1.6 }}>
          ⚠️ 本页用 add_advisor 模型对<b>全 A 股</b>按预测 {horizon} 日收益做<b>横截面相对排名</b>并分桶。
          edge 很薄 (rank-IC≈0.05),这是<b>相对强弱排名, 不是绝对涨跌保证</b>;仅 A 股, 未扣交易成本, 请配合止盈/止损与仓位纪律。
        </div>

        <Card style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 16 }} styles={{ body: { padding: 16 } }}>
          <Row gutter={16} align="middle">
            <Col span={6}>
              <Button type="primary" size="large" onClick={runScan} loading={loading} block icon={<ReloadOutlined />} style={{ background: GOLD, borderColor: GOLD }}>
                手动触发选股扫描
              </Button>
            </Col>
            <Col span={18}>
              <p style={{ margin: 0, color: 'rgba(255,255,255,0.6)', fontSize: 13, lineHeight: 1.6 }}>
                点击后调用 <code style={{ color: GOLD }}>/api/advisor/scan?refresh=1</code> 强制重算,
                列出预测 {horizon} 日收益排名靠前(买入档)与靠后(卖出档)的股票。全 A 股扫描可能耗时数十秒。
                {data && (
                  <span style={{ display: 'block', marginTop: 6, color: 'rgba(255,255,255,0.45)', fontSize: 12 }}>
                    数据日 {data.predDate || '—'} · 参与 {data.totalStocks ?? 0} 只 · 生成于 {data.generatedAt || '—'}
                    {data.cached ? ` · 缓存(${data.cacheAgeMin ?? 0}分钟前)` : ' · 实时重算'}
                  </span>
                )}
              </p>
            </Col>
          </Row>
        </Card>

        {loading && <Spin tip="add_advisor 全 A 股截面打分中..." style={{ display: 'block', margin: '60px auto' }} />}

        {!loading && data && (
          <>
            {/* 分布概览 */}
            {data.distribution && (
              <Card style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 16 }} styles={{ body: { padding: 16 } }}>
                <Row gutter={16}>
                  {([
                    ['strong_buy', '强烈买入', '#3fb950'],
                    ['buy', '买入', '#7bc96f'],
                    ['hold', '持有', 'rgba(255,255,255,0.6)'],
                    ['sell', '卖出', '#faad14'],
                    ['strong_sell', '强烈卖出', '#f85149'],
                  ] as [string, string, string][]).map(([k, label, color]) => (
                    <Col span={4} key={k}>
                      <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12 }}>{label}</div>
                      <div style={{ color, fontSize: 24, fontWeight: 700 }}>{data.distribution?.[k] ?? 0}</div>
                    </Col>
                  ))}
                </Row>
              </Card>
            )}

            {/* 买入档 top 预测收益柱状 */}
            {topBuys.length > 0 && (
              <Card title={<span style={{ color: 'rgba(255,255,255,0.85)' }}>买入档 · 预测{horizon}日收益 (前 {topBuys.length})</span>} style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 16 }}>
                <div style={{ height: 300 }}>
                  <Bar data={{
                    labels: topBuys.map(s => s.name),
                    datasets: [
                      { label: `预测${horizon}日收益(%)`, data: topBuys.map(s => +(s.score * 100).toFixed(2)), backgroundColor: 'rgba(63,185,80,0.6)' },
                      { label: '上涨概率(%)', data: topBuys.map(s => +(s.upProb * 100).toFixed(1)), backgroundColor: 'rgba(24,144,255,0.4)' },
                    ],
                  }} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: darkLegend }, scales: { x: { ticks: darkTicks }, y: { ticks: darkTicks } } }} />
                </div>
              </Card>
            )}

            {/* 买入榜 */}
            <Card title={<span style={{ color: '#3fb950' }}>买入榜 ({buys.length})</span>} style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 16 }}>
              <Table columns={columns} dataSource={buys} rowKey="symbol" pagination={{ pageSize: 20 }} size="small" scroll={{ x: 780 }} />
            </Card>

            {/* 卖出/回避榜 */}
            <Card title={<span style={{ color: '#f85149' }}>卖出 / 回避榜 ({sells.length})</span>} style={{ background: '#242830', border: '1px solid #3a3f4a' }}>
              <Table columns={columns} dataSource={sells} rowKey="symbol" pagination={{ pageSize: 20 }} size="small" scroll={{ x: 780 }} />
            </Card>
          </>
        )}

        {!loading && !data && (
          <Card style={{ textAlign: 'center', padding: 60, background: '#242830', border: '1px solid #3a3f4a' }}>
            <StockOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
            <p style={{ marginTop: 16, color: '#999' }}>点击上方「手动触发选股扫描」开始</p>
          </Card>
        )}
      </div>
    </div>
  );
}
