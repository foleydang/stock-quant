import { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler,
} from 'chart.js';
import { Card, Statistic, Row, Col, Tag, Spin } from 'antd';
import { RiseOutlined, FallOutlined } from '@ant-design/icons';
import axios from 'axios';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler);

const CARD_BG = '#242830';
const CARD_BORDER = '#3a3f4a';
const GOLD = '#e2b04a';
const TEXT_DIM = 'rgba(255,255,255,0.5)';
const TEXT_LIGHT = 'rgba(255,255,255,0.85)';

interface CurrentSignal {
  dataDate: string;
  lastPrice: number;
  rsi: number;
  candidate: boolean;
  ret20Pred: number;
  upProb: number;
  tpProb: number;
  tpPrice: number;
  slPrice: number;
  verdict: string;
}
interface OosSeriesPt { date: string; pred: number; actual: number; }
interface Oos {
  n: number;
  dir_acc: number;
  hit_rate_up: number | null;
  mean_ret_up_net: number | null;
  series: OosSeriesPt[];
}
interface PredictData {
  status: string;
  symbol: string;
  horizon: number;
  current: CurrentSignal | null;
  oos: Oos | null;
  caveat: string;
}

const darkChartPlugin = {
  id: 'darkChartBg',
  beforeDraw: (chart: any) => {
    const ctx = chart.ctx;
    ctx.save();
    ctx.globalCompositeOperation = 'destination-over';
    ctx.fillStyle = CARD_BG;
    ctx.fillRect(0, 0, chart.width, chart.height);
    ctx.restore();
  },
};

interface Props { symbol: string; }

export default function Forecast7Tab({ symbol }: Props) {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<PredictData | null>(null);
  const [msg, setMsg] = useState<string>('');

  useEffect(() => { fetchData(); }, [symbol]);

  const fetchData = async () => {
    setLoading(true);
    setMsg('');
    try {
      const res = await axios.get(`/api/advisor/predict/${symbol}`);
      if (res.data.status === 'success') setData(res.data);
      else { setData(null); setMsg(res.data.message || '暂无预测'); }
    } catch {
      setData(null);
      setMsg('无法连接到服务器');
    }
    setLoading(false);
  };

  if (loading) return <Spin tip="模型预测中..." style={{ display: 'block', margin: '60px auto' }} />;
  if (!data) return <Card style={{ textAlign: 'center', padding: 40, background: CARD_BG, border: `1px solid ${CARD_BORDER}` }}><p style={{ color: TEXT_DIM }}>{msg || '暂无预测数据'}</p></Card>;

  const c = data.current;
  const oos = data.oos;

  const verdictTag = (v: string) => {
    if (v.includes('补') && !v.includes('不')) return <Tag color="green" style={{ fontSize: 14 }}><RiseOutlined /> {v}</Tag>;
    if (v.includes('不补') || v.includes('回避') || v.includes('减')) return <Tag color="red" style={{ fontSize: 14 }}><FallOutlined /> {v}</Tag>;
    return <Tag color="gold" style={{ fontSize: 14 }}>{v}</Tag>;
  };

  // OOS 预测 vs 实际 (月度采样序列)
  const oosChart = oos && oos.series.length ? {
    labels: oos.series.map((p) => p.date.slice(2)),
    datasets: [
      { label: '预测20日收益(%)', data: oos.series.map((p) => +(p.pred * 100).toFixed(2)), borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.08)', borderWidth: 2, pointRadius: 2, tension: 0.15, fill: true },
      { label: '实际20日收益(%)', data: oos.series.map((p) => +(p.actual * 100).toFixed(2)), borderColor: GOLD, borderWidth: 2, pointRadius: 2, tension: 0.15 },
      { label: '0 轴', data: oos.series.map(() => 0), borderColor: 'rgba(255,255,255,0.25)', borderWidth: 1, borderDash: [3, 3], pointRadius: 0 },
    ],
  } : null;

  const oosOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { title: { display: true, text: `${symbol} 样本外(OOS) 预测 vs 实际 · 20日收益`, color: TEXT_LIGHT, font: { size: 14 } }, legend: { labels: { color: TEXT_DIM } } },
    scales: { x: { ticks: { color: TEXT_DIM, maxTicksLimit: 14 }, grid: { color: CARD_BORDER } }, y: { ticks: { color: TEXT_DIM, callback: (v: any) => v + '%' }, grid: { color: CARD_BORDER } } },
  };

  return (
    <div>
      {/* 诚实 caveat */}
      <div style={{ background: 'rgba(226,176,74,0.08)', border: `1px solid ${GOLD}`, borderRadius: 6, padding: '10px 14px', marginBottom: 16, color: GOLD, fontSize: 13, lineHeight: 1.6 }}>
        ⚠️ {data.caveat}
      </div>

      {/* 当前 20 日信号 */}
      {c ? (
        <Card title={<span style={{ color: TEXT_LIGHT }}>当前 {data.horizon} 日预测信号 · 数据日 {c.dataDate}</span>} style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}`, marginBottom: 16 }} styles={{ body: { padding: 14 } }}>
          <Row gutter={16}>
            <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>当前价</span>} value={c.lastPrice} prefix="¥" precision={2} valueStyle={{ color: TEXT_LIGHT, fontSize: 22 }} /></Col>
            <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>预测20日收益</span>} value={c.ret20Pred * 100} suffix="%" precision={2} valueStyle={{ color: c.ret20Pred >= 0 ? '#3fb950' : '#f85149', fontSize: 22 }} prefix={c.ret20Pred >= 0 ? <RiseOutlined /> : <FallOutlined />} /></Col>
            <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>上涨概率</span>} value={c.upProb * 100} suffix="%" precision={1} valueStyle={{ color: c.upProb >= 0.5 ? '#3fb950' : '#f85149', fontSize: 22 }} /></Col>
            <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>RSI14</span>} value={c.rsi} precision={0} valueStyle={{ color: c.rsi < 40 ? '#3fb950' : c.rsi > 70 ? '#f85149' : TEXT_LIGHT, fontSize: 22 }} /></Col>
            <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>止盈位</span>} value={c.tpPrice} prefix="¥" precision={2} valueStyle={{ color: '#3fb950', fontSize: 22 }} /></Col>
            <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>止损位</span>} value={c.slPrice} prefix="¥" precision={2} valueStyle={{ color: '#f85149', fontSize: 22 }} /></Col>
            <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>补仓候选态</span>} valueRender={() => c.candidate ? <Tag color="gold">候选(跌破MA20+超卖)</Tag> : <Tag>非候选</Tag>} /></Col>
            <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>建议</span>} valueRender={() => verdictTag(c.verdict)} /></Col>
          </Row>
        </Card>
      ) : (
        <Card style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}`, marginBottom: 16, textAlign: 'center', padding: 20 }}>
          <span style={{ color: TEXT_DIM }}>该股数据不足, 无法出当前信号</span>
        </Card>
      )}

      {/* OOS 历史命中率 + 预测vs实际 */}
      {oos ? (
        <>
          <Card style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}`, marginBottom: 16 }} styles={{ body: { padding: 14 } }}>
            <Row gutter={16}>
              <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM }}>OOS 样本数</span>} value={oos.n} valueStyle={{ color: TEXT_LIGHT, fontSize: 22 }} /></Col>
              <Col span={5}><Statistic title={<span style={{ color: TEXT_DIM }}>方向准确率</span>} value={oos.dir_acc * 100} suffix="%" precision={1} valueStyle={{ color: oos.dir_acc >= 0.5 ? '#3fb950' : '#f85149', fontSize: 22 }} /></Col>
              <Col span={5}><Statistic title={<span style={{ color: TEXT_DIM }}>预测涨时实际上涨率</span>} value={oos.hit_rate_up != null ? oos.hit_rate_up * 100 : 0} suffix="%" precision={1} valueStyle={{ color: (oos.hit_rate_up ?? 0) >= 0.5 ? '#3fb950' : '#f85149', fontSize: 22 }} /></Col>
              <Col span={6}><Statistic title={<span style={{ color: TEXT_DIM }}>预测涨时平均净收益(扣成本)</span>} value={oos.mean_ret_up_net != null ? oos.mean_ret_up_net * 100 : 0} suffix="%" precision={2} valueStyle={{ color: (oos.mean_ret_up_net ?? 0) >= 0 ? '#3fb950' : '#f85149', fontSize: 22 }} /></Col>
            </Row>
          </Card>
          {oosChart && (
            <Card style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 12 } }}>
              <div style={{ height: 320 }}><Line data={oosChart} options={oosOpts} plugins={[darkChartPlugin]} /></div>
            </Card>
          )}
        </>
      ) : (
        <Card style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}`, textAlign: 'center', padding: 20 }}>
          <span style={{ color: TEXT_DIM }}>该股不在回测池 (仅A股池有 OOS 历史), 无预测vs实际记录</span>
        </Card>
      )}
    </div>
  );
}
