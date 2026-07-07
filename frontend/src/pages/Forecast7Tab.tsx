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
interface PredictData {
  status: string;
  symbol: string;
  horizon: number;
  current: CurrentSignal | null;
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

  const verdictTag = (v: string) => {
    if (v.includes('补') && !v.includes('不')) return <Tag color="green" style={{ fontSize: 14 }}><RiseOutlined /> {v}</Tag>;
    if (v.includes('不补') || v.includes('回避') || v.includes('减')) return <Tag color="red" style={{ fontSize: 14 }}><FallOutlined /> {v}</Tag>;
    return <Tag color="gold" style={{ fontSize: 14 }}>{v}</Tag>;
  };

  // 未来 HORIZON 个交易日的预测: 模型只给累计终点(目标价), 逐日波动无法诚实预测
  // 画一条"到目标价的隐含平均漂移"直线 + 止盈/止损参考线, 只示意方向
  const days = data.horizon;
  const target = c ? c.lastPrice * (1 + c.ret20Pred) : 0;
  const up = c ? c.ret20Pred >= 0 : true;
  const fwdChart = c ? {
    labels: Array.from({ length: days + 1 }, (_, i) => (i === 0 ? '今日' : `T+${i}`)),
    datasets: [
      { label: '预测路径(隐含平均漂移)', data: Array.from({ length: days + 1 }, (_, i) => +(c.lastPrice + (target - c.lastPrice) * i / days).toFixed(2)), borderColor: up ? '#3fb950' : '#f85149', backgroundColor: up ? 'rgba(63,185,80,0.10)' : 'rgba(248,81,73,0.10)', borderWidth: 2, pointRadius: 0, tension: 0, fill: true },
      { label: '止盈位', data: Array.from({ length: days + 1 }, () => c.tpPrice), borderColor: '#3fb950', borderWidth: 1, borderDash: [5, 4], pointRadius: 0 },
      { label: '止损位', data: Array.from({ length: days + 1 }, () => c.slPrice), borderColor: '#f85149', borderWidth: 1, borderDash: [5, 4], pointRadius: 0 },
    ],
  } : null;

  const fwdOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      title: { display: true, text: `${symbol} 未来 ${days} 交易日预测 · 目标价 ¥${target.toFixed(2)} (${(c ? c.ret20Pred * 100 : 0).toFixed(2)}%)`, color: TEXT_LIGHT, font: { size: 14 } },
      legend: { labels: { color: TEXT_DIM } },
    },
    scales: { x: { ticks: { color: TEXT_DIM, maxTicksLimit: 21 }, grid: { color: CARD_BORDER } }, y: { ticks: { color: TEXT_DIM, callback: (v: any) => '¥' + v }, grid: { color: CARD_BORDER } } },
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

      {/* 未来 N 交易日预测走势 (前瞻, 不含历史) */}
      {fwdChart && (
        <Card style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 12 } }}>
          <div style={{ height: 320 }}><Line data={fwdChart} options={fwdOpts} plugins={[darkChartPlugin]} /></div>
          <div style={{ color: TEXT_DIM, fontSize: 12, marginTop: 10, lineHeight: 1.7 }}>
            ⚠️ 模型只预测<b style={{ color: GOLD }}>未来 {days} 个交易日的累计涨跌</b>(方向 + 幅度 + 概率), <b>并不预测每一天的具体涨跌</b>。
            上图直线是"到达目标价的隐含平均漂移", 仅示意方向与目标位; 真实走势必然上下波动。
            逐日涨跌路径无法诚实预测(编造的逐日价格已删除)。请结合止盈/止损位与纪律执行。
          </div>
        </Card>
      )}
    </div>
  );
}
