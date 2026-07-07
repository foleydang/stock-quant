import { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Spin, Table, TableColumnsType } from 'antd';
import { RiseOutlined, FallOutlined } from '@ant-design/icons';
import { Line, Bar } from 'react-chartjs-2';
import axios from 'axios';

const CARD_BG = '#242830';
const CARD_BORDER = '#3a3f4a';
const GOLD = '#e2b04a';
const TEXT_DIM = 'rgba(255,255,255,0.5)';

interface CurvePt { date: string; value: number; }
interface Strategy {
  curve: CurvePt[];
  total_return: number;
  annual_return: number;
  sharpe: number;
  max_drawdown: number;
  win_rate: number;
  avg_period_ret: number;
  n_periods: number;
}
interface YearRow { year: number; top_avg: number; ls_avg: number; uni_avg: number; n_periods: number; }
interface SingleName {
  timing_net_per_trade: number; buyhold_net_per_trade: number;
  timing_edge_per_trade: number; n_trades: number; n_timing_trades: number;
}
interface Backtest {
  status: string;
  strategies: Record<string, Strategy>;
  single_name: SingleName;
  by_year: YearRow[];
  headline: {
    top_k_total: number; universe_total: number; excess_total: number;
    long_short_total: number; long_short_sharpe: number; top_k_excess_per_period: number;
    top_k_annual: number; top_k_sharpe: number; top_k_maxdd: number;
    single_name_timing_edge: number;
  };
  config: {
    horizon: number; top_quantile: number; bot_quantile: number;
    cost_roundtrip: number; rebalance_days: number; n_rebalances: number;
    span: string[];
  };
  caveat: string;
  generated_at: string;
}

const pct = (x: number) => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(1)}%`;
const col = (x: number) => (x >= 0 ? '#52c41a' : '#ff4d4f');

// 每 4 个点采样一次控制曲线密度
const sample = <T,>(arr: T[], step: number) => arr.filter((_, i) => i % step === 0);

export default function PortfolioBacktest() {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<Backtest | null>(null);
  const [err, setErr] = useState<string>('');

  useEffect(() => { fetchData(); }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await axios.get('/api/advisor/backtest');
      if (res.data.status === 'success') setData(res.data);
      else setErr(res.data.message || '回测结果未就绪');
    } catch (e) { setErr('无法连接到服务器'); }
    setLoading(false);
  };

  if (loading) return <Spin tip="加载回测结果..." style={{ display: 'block', margin: '60px auto' }} />;
  if (!data) return (
    <Card style={{ textAlign: 'center', padding: 40, background: CARD_BG, border: `1px solid ${CARD_BORDER}` }}>
      <p style={{ color: TEXT_DIM }}>{err || '暂无回测数据'}</p>
    </Card>
  );

  const h = data.headline;
  const s = data.strategies;
  const step = Math.max(1, Math.floor((s.top_k?.curve.length || 1) / 120));

  const labels = sample(s.universe?.curve || [], step).map((p) => p.date);
  const navChart = {
    labels,
    datasets: [
      { label: '横截面 Top-K (做多预测高分档)', data: sample(s.top_k?.curve || [], step).map((p) => p.value), borderColor: '#52c41a', backgroundColor: 'rgba(82,196,26,0.08)', borderWidth: 2, pointRadius: 0, tension: 0.1, fill: true },
      { label: 'Long-Short (市场中性)', data: sample(s.long_short?.curve || [], step).map((p) => p.value), borderColor: GOLD, borderWidth: 2, pointRadius: 0, tension: 0.1 },
      { label: '全市场等权 (基准)', data: sample(s.universe?.curve || [], step).map((p) => p.value), borderColor: 'rgba(255,255,255,0.45)', borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, tension: 0.1 },
    ],
  };

  const candChart = {
    labels: sample(s.candidate_a3?.curve || [], step).map((p) => p.date),
    datasets: [
      { label: '方案3 候选态 (跌破MA20+超卖) top-decile', data: sample(s.candidate_a3?.curve || [], step).map((p) => p.value), borderColor: '#13c2c2', backgroundColor: 'rgba(19,194,194,0.08)', borderWidth: 2, pointRadius: 0, tension: 0.1, fill: true },
    ],
  };

  const yearChart = {
    labels: data.by_year.map((y) => String(y.year)),
    datasets: [
      { label: 'Top-K 平均每期净收益', data: data.by_year.map((y) => y.top_avg * 100), backgroundColor: 'rgba(82,196,26,0.6)' },
      { label: 'Long-Short', data: data.by_year.map((y) => y.ls_avg * 100), backgroundColor: 'rgba(226,176,74,0.6)' },
      { label: '基准', data: data.by_year.map((y) => y.uni_avg * 100), backgroundColor: 'rgba(255,255,255,0.35)' },
    ],
  };

  const lineOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: 'rgba(255,255,255,0.7)', boxWidth: 14 } } },
    scales: {
      x: { ticks: { color: 'rgba(255,255,255,0.4)', maxTicksLimit: 12 }, grid: { color: '#2f333c' } },
      y: { ticks: { color: 'rgba(255,255,255,0.4)' }, grid: { color: '#2f333c' } },
    },
  };
  const barOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: 'rgba(255,255,255,0.7)', boxWidth: 14 } } },
    scales: {
      x: { ticks: { color: 'rgba(255,255,255,0.4)' }, grid: { color: '#2f333c' } },
      y: { ticks: { color: 'rgba(255,255,255,0.4)', callback: (v: any) => v + '%' }, grid: { color: '#2f333c' } },
    },
  };

  const yearCols: TableColumnsType<YearRow> = [
    { title: '年份', dataIndex: 'year', width: 80 },
    { title: 'Top-K 均值/期', dataIndex: 'top_avg', render: (x: number) => <span style={{ color: col(x) }}>{pct(x)}</span> },
    { title: 'Long-Short', dataIndex: 'ls_avg', render: (x: number) => <span style={{ color: col(x) }}>{pct(x)}</span> },
    { title: '基准', dataIndex: 'uni_avg', render: (x: number) => <span style={{ color: col(x) }}>{pct(x)}</span> },
    { title: '期数', dataIndex: 'n_periods', width: 70 },
  ];

  return (
    <div>
      {/* 诚实 caveat */}
      <div style={{ background: 'rgba(226,176,74,0.08)', border: `1px solid ${GOLD}`, borderRadius: 6, padding: '10px 14px', marginBottom: 16, color: GOLD, fontSize: 13, lineHeight: 1.6 }}>
        ⚠️ {data.caveat}
        <div style={{ color: TEXT_DIM, marginTop: 4, fontSize: 12 }}>
          样本外 walk-forward · {data.config.span?.[0]} ~ {data.config.span?.[1]} · 不重叠 {data.config.rebalance_days} 交易日 rebalance · {data.config.n_rebalances} 期 · 生成于 {data.generated_at}
        </div>
      </div>

      {/* headline 指标 */}
      <Card style={{ marginBottom: 16, background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 14 } }}>
        <Row gutter={16}>
          <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM }}>Top-K 总收益</span>} value={h.top_k_total * 100} precision={1} suffix="%" valueStyle={{ color: col(h.top_k_total), fontSize: 24 }} prefix={<RiseOutlined />} /></Col>
          <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM }}>基准总收益</span>} value={h.universe_total * 100} precision={1} suffix="%" valueStyle={{ color: 'rgba(255,255,255,0.75)', fontSize: 24 }} /></Col>
          <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM }}>超额收益</span>} value={h.excess_total * 100} precision={1} suffix="%" valueStyle={{ color: col(h.excess_total), fontSize: 24 }} prefix={h.excess_total >= 0 ? <RiseOutlined /> : <FallOutlined />} /></Col>
          <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM }}>Long-Short 总收益</span>} value={h.long_short_total * 100} precision={1} suffix="%" valueStyle={{ color: col(h.long_short_total), fontSize: 24 }} /></Col>
          <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM }}>Top-K 年化</span>} value={h.top_k_annual * 100} precision={1} suffix="%" valueStyle={{ color: col(h.top_k_annual), fontSize: 24 }} /></Col>
          <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM }}>Top-K Sharpe</span>} value={h.top_k_sharpe} precision={2} valueStyle={{ color: h.top_k_sharpe >= 1 ? '#52c41a' : '#faad14', fontSize: 24 }} /></Col>
        </Row>
        <Row gutter={16} style={{ marginTop: 10 }}>
          <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM }}>Top-K 最大回撤</span>} value={h.top_k_maxdd * 100} precision={1} suffix="%" valueStyle={{ color: '#ff7a45', fontSize: 20 }} /></Col>
          <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM }}>Top-K 胜率</span>} value={(s.top_k?.win_rate || 0) * 100} precision={0} suffix="%" valueStyle={{ color: 'rgba(255,255,255,0.75)', fontSize: 20 }} /></Col>
          <Col span={5}><Statistic title={<span style={{ color: TEXT_DIM }}>每期超额 (top-K − 基准)</span>} value={h.top_k_excess_per_period * 100} precision={2} suffix="%" valueStyle={{ color: col(h.top_k_excess_per_period), fontSize: 20 }} /></Col>
          <Col span={5}><Statistic title={<span style={{ color: TEXT_DIM }}>Long-Short Sharpe (最可信)</span>} value={h.long_short_sharpe} precision={2} valueStyle={{ color: h.long_short_sharpe >= 1 ? '#52c41a' : '#faad14', fontSize: 20 }} /></Col>
          <Col span={6}>
            <Statistic
              title={<span style={{ color: TEXT_DIM }}>诚实对照: 单只择时逐笔 − 买入持有 (≈0 = 无择时edge)</span>}
              value={h.single_name_timing_edge * 100} precision={2} suffix="%"
              valueStyle={{ color: Math.abs(h.single_name_timing_edge) < 0.005 ? '#faad14' : col(h.single_name_timing_edge), fontSize: 20 }}
            />
          </Col>
        </Row>
      </Card>

      {/* 净值曲线 */}
      <Card title={<span style={{ color: '#52c41a' }}>组合净值曲线 (基准起点 = 1.0, 扣成本)</span>} style={{ marginBottom: 16, background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 12 } }}>
        <div style={{ height: 320 }}><Line data={navChart} options={lineOpts} /></div>
      </Card>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          <Card title={<span style={{ color: '#ff7a45' }}>诚实对照: 单只择时逐笔收益 (非组合)</span>} size="small" style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 16 } }}>
            <Row gutter={16}>
              <Col span={12}><Statistic title={<span style={{ color: TEXT_DIM }}>预测涨才买·平均每笔(扣成本)</span>} value={data.single_name.timing_net_per_trade * 100} precision={2} suffix="%" valueStyle={{ color: 'rgba(255,255,255,0.85)', fontSize: 22 }} /></Col>
              <Col span={12}><Statistic title={<span style={{ color: TEXT_DIM }}>无差别买入持有·平均每笔</span>} value={data.single_name.buyhold_net_per_trade * 100} precision={2} suffix="%" valueStyle={{ color: 'rgba(255,255,255,0.85)', fontSize: 22 }} /></Col>
            </Row>
            <div style={{ color: '#faad14', fontSize: 12, marginTop: 12, lineHeight: 1.6 }}>
              两者≈打平(差 {(data.single_name.timing_edge_per_trade * 100).toFixed(2)}%),说明<b>单只择时没有 edge</b>。
              模型的价值在<b>横截面排序</b>(上方 top-K 超额 / long-short),不是"单只该不该买"。
              共 {data.single_name.n_trades.toLocaleString()} 笔样本。
            </div>
          </Card>
        </Col>
        <Col span={12}>
          <Card title={<span style={{ color: '#13c2c2' }}>方案3 候选态 (补仓机会) 净值</span>} size="small" style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 12 } }}>
            <div style={{ height: 260 }}><Line data={candChart} options={lineOpts} /></div>
          </Card>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={14}>
          <Card title="分年平均每期净收益" size="small" style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 12 } }}>
            <div style={{ height: 280 }}><Bar data={yearChart} options={barOpts} /></div>
          </Card>
        </Col>
        <Col span={10}>
          <Card title="分年明细" size="small" style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 12 } }}>
            <Table columns={yearCols} dataSource={data.by_year} rowKey="year" pagination={false} size="small" />
          </Card>
        </Col>
      </Row>
    </div>
  );
}
