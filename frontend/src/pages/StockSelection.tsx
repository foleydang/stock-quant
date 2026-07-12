import { useState, useEffect } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend,
} from 'chart.js';
import { Button, Card, Row, Col, Tag, Table, Spin, message, Select } from 'antd';
import { StockOutlined, ReloadOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import axios from 'axios';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const GOLD = '#e2b04a';
const BOARDS = [
  { value: 'all', label: '全部 A 股' },
  { value: 'sh', label: '上海主板 (600/601/603/605)' },
  { value: 'sz', label: '深圳主板 (000/001/002/003)' },
  { value: 'cyb', label: '创业板 (300/301)' },
  { value: 'kcb', label: '科创板 (688)' },
];

interface ScanItem {
  rank: number;
  symbol: string;
  name: string;
  score: number;
  signal: string;
  upProb: number;
  tpProb: number;
  candidate: boolean;
}
interface ScanPayload {
  status: string;
  predDate?: string;
  totalStocks?: number;
  distribution?: Record<string, number>;
  signals?: { strong_buy?: ScanItem[]; buy?: ScanItem[]; hold?: ScanItem[]; sell?: ScanItem[]; strong_sell?: ScanItem[] };
  horizon?: number;
  caveat?: string;
  generatedAt?: string;
  cached?: boolean;
  cacheAgeMin?: number;
  board?: string;
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
  const [board, setBoard] = useState('all');
  const [scanning, setScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState('');

  useEffect(() => { fetchScan(false); }, []);

  // 轮询扫描状态
  useEffect(() => {
    if (!scanning) return;
    const timer = setInterval(async () => {
      try {
        const res = await axios.get('/api/advisor/scan/status');
        if (!res.data.scanning) {
          setScanning(false);
          setScanProgress('');
          fetchScan(false);  // 刷新数据
          message.success('扫描完成');
        } else {
          setScanProgress(res.data.progress || '');
        }
      } catch {
        // ignore
      }
    }, 3000);
    return () => clearInterval(timer);
  }, [scanning]);

  const fetchScan = async (refresh: boolean) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ board, limit: '100' });
      if (refresh) params.set('refresh', '1');
      const res = await axios.get('/api/advisor/scan?' + params.toString());
      if (res.data.status === 'success' || res.data.status === 'scanning') {
        setData(res.data);
        if (res.data.scanning) {
          setScanning(true);
          message.info('后台扫描已启动, 请稍候...');
        } else if (refresh) {
          message.success(`扫描完成, 共 ${res.data.totalStocks ?? 0} 只`);
        } else if (res.data.cached) {
          // 静默加载缓存
        } else {
          message.success(`扫描完成, 共 ${res.data.totalStocks ?? 0} 只`);
        }
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

  const runScan = () => fetchScan(true);

  const horizon = data?.horizon ?? 20;
  const buys: ScanItem[] = [...(data?.signals?.strong_buy || []), ...(data?.signals?.buy || [])];
  const sells: ScanItem[] = [...(data?.signals?.sell || []), ...(data?.signals?.strong_sell || [])];
  const holds: ScanItem[] = data?.signals?.hold || [];
  const topBuys = buys.slice(0, 15);

  const columns = [
    { title: '排名', dataIndex: 'rank', width: 60 },
    { title: '股票', dataIndex: 'name', width: 120 },
    { title: '代码', dataIndex: 'symbol', width: 110 },
    { title: '信号', dataIndex: 'signal', width: 100, render: (v: string) => signalTag(v) },
    { title: `预测${horizon}日收益`, dataIndex: 'score', width: 120, render: (r: number) => <span style={{ color: r >= 0 ? '#3fb950' : '#f85149', fontWeight: 700 }}>{r >= 0 ? '+' : ''}{(r * 100).toFixed(2)}%</span> },
    { title: '上涨概率', dataIndex: 'upProb', width: 90, render: (p: number) => <span style={{ color: p >= 0.5 ? '#3fb950' : 'rgba(255,255,255,0.7)' }}>{(p * 100).toFixed(1)}%</span> },
    { title: '止盈概率', dataIndex: 'tpProb', width: 90, render: (p: number) => <span style={{ color: 'rgba(255,255,255,0.7)' }}>{(p * 100).toFixed(1)}%</span> },
    { title: '补仓候选', dataIndex: 'candidate', width: 90, render: (c: boolean) => c ? <Tag color="gold">候选</Tag> : <Tag>—</Tag> },
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
          ⚠️ 本页用 add_advisor 模型对<b>选定板块</b>按预测 {horizon} 日收益做<b>横截面相对排名</b>并分桶。
          edge 很薄 (rank-IC≈0.05),这是<b>相对强弱排名, 不是绝对涨跌保证</b>;仅 A 股, 未扣交易成本, 请配合止盈/止损与仓位纪律。
        </div>

        <Card style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 16 }} styles={{ body: { padding: 16 } }}>
          <Row gutter={16} align="middle">
            <Col span={5}>
              <Select
                value={board}
                onChange={(v) => setBoard(v)}
                options={BOARDS}
                style={{ width: '100%' }}
                popupMatchSelectWidth={false}
              />
            </Col>
            <Col span={5}>
              <Button type="primary" size="large" onClick={runScan} loading={loading} block icon={<ReloadOutlined />} style={{ background: GOLD, borderColor: GOLD, height: 40 }}>
                手动触发扫描
              </Button>
            </Col>
            <Col span={14}>
              <p style={{ margin: 0, color: 'rgba(255,255,255,0.6)', fontSize: 13, lineHeight: 1.6 }}>
                默认扫描 {board === 'all' ? '全部 A 股' : BOARDS.find(b => b.value === board)?.label} 中数据量最多的 100 只。
                {data ? (
                  <span style={{ display: 'block', marginTop: 6, color: 'rgba(255,255,255,0.45)', fontSize: 12 }}>
                    数据日 {data.predDate || '—'} · 参与 {data.totalStocks ?? 0} 只 · 生成于 {data.generatedAt || '—'}
                    {data.cached ? ` · 缓存(${data.cacheAgeMin ?? 0}分钟前)` : ' · 实时'}
                  </span>
                ) : (
                  <span style={{ display: 'block', marginTop: 6, color: 'rgba(255,255,255,0.35)', fontSize: 12 }}>
                    点击「手动触发扫描」或等待自动加载缓存
                  </span>
                )}
              </p>
            </Col>
          </Row>
        </Card>

        {loading && <Spin tip="add_advisor 截面打分中..." style={{ display: 'block', margin: '60px auto' }} />}

        {scanning && (
          <div style={{ background: 'rgba(24,144,255,0.1)', border: '1px solid rgba(24,144,255,0.3)', borderRadius: 6, padding: '12px 16px', marginBottom: 16, color: 'rgba(255,255,255,0.8)', fontSize: 13, display: 'flex', alignItems: 'center', gap: 10 }}>
            <Spin size="small" />
            <span>后台扫描中{scanProgress ? ` (${scanProgress} 批)` : ''}... 完成后自动刷新</span>
          </div>
        )}

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

            {/* 买入档 预测收益柱状图 */}
            {topBuys.length > 0 && (
              <Card title={<span style={{ color: 'rgba(255,255,255,0.85)' }}>买入档 · 预测{horizon}日收益 Top {topBuys.length}</span>} style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 16 }}>
                <div style={{ height: 300 }}>
                  <Bar data={{
                    labels: topBuys.map(s => s.name),
                    datasets: [
                      { label: `预测${horizon}日收益(%)`, data: topBuys.map(s => +(s.score * 100).toFixed(2)), backgroundColor: topBuys.map(s => s.score >= 0.03 ? 'rgba(63,185,80,0.7)' : 'rgba(63,185,80,0.4)') },
                    ],
                  }} options={{
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: darkLegend, tooltip: { callbacks: { label: (ctx: any) => `${ctx.dataset.label}: ${ctx.raw}%` } } },
                    scales: { x: { ticks: { ...darkTicks, maxRotation: 45 } }, y: { ticks: { ...darkTicks, callback: (v: any) => v + '%' } } },
                  }} />
                </div>
              </Card>
            )}

            {/* 买入榜 */}
            <Card title={<span style={{ color: '#3fb950' }}>🟢 买入榜 ({buys.length})</span>} style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 16 }}>
              <Table columns={columns} dataSource={buys} rowKey="symbol" pagination={{ pageSize: 20 }} size="small" scroll={{ x: 780 }} />
            </Card>

            {/* 持有榜 */}
            {holds.length > 0 && (
              <Card title={<span style={{ color: 'rgba(255,255,255,0.7)' }}>🟡 持有档 ({holds.length})</span>} style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 16 }}>
                <Table columns={columns} dataSource={holds} rowKey="symbol" pagination={{ pageSize: 20 }} size="small" scroll={{ x: 780 }} />
              </Card>
            )}

            {/* 卖出/回避榜 */}
            <Card title={<span style={{ color: '#f85149' }}>🔴 卖出 / 回避榜 ({sells.length})</span>} style={{ background: '#242830', border: '1px solid #3a3f4a' }}>
              <Table columns={columns} dataSource={sells} rowKey="symbol" pagination={{ pageSize: 20 }} size="small" scroll={{ x: 780 }} />
            </Card>
          </>
        )}

        {!loading && !data && (
          <Card style={{ textAlign: 'center', padding: 60, background: '#242830', border: '1px solid #3a3f4a' }}>
            <StockOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
            <p style={{ marginTop: 16, color: '#999' }}>选择板块后点击「手动触发扫描」开始</p>
          </Card>
        )}
      </div>
    </div>
  );
}