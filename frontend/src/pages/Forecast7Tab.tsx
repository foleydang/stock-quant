import { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler,
} from 'chart.js';
import { Card, Statistic, Row, Col, Tag, Spin, Table, Progress, TableColumnsType } from 'antd';
import { RiseOutlined, FallOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import axios from 'axios';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler);

const CARD_BG = '#242830';
const CARD_BORDER = '#3a3f4a';
const GOLD = '#e2b04a';
const TEXT_DIM = 'rgba(255,255,255,0.5)';
const TEXT_LIGHT = 'rgba(255,255,255,0.85)';

interface Prediction7Day {
  day: number;
  date: string;
  upProb: number;
  downProb: number;
  direction: string;
  simPrice: number;
  priceLow: number;
  priceHigh: number;
}

interface Forecast7Data {
  symbol: string;
  stockName: string;
  currentPrice: number;
  lastDate: string;
  predictions: Prediction7Day[];
  summary: {
    avgUpProb: number;
    trendDirection: string;
    simFinalPrice: number;
    simReturn: number;
  };
}

interface HistoryDay {
  date: string;
  predCount: number;
  correctCount: number;
  accuracy: number;
  avgUpProb: number;
}

interface HistoryData {
  symbol: string;
  stockName: string;
  days: number;
  overallAccuracy: number;
  totalPredictions: number;
  totalCorrect: number;
  dailyRecords: HistoryDay[];
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

interface Props {
  symbol: string;
}

export default function Forecast7Tab({ symbol }: Props) {
  const [loading, setLoading] = useState(false);
  const [data7d, setData7d] = useState<Forecast7Data | null>(null);
  const [historyData, setHistoryData] = useState<HistoryData | null>(null);

  useEffect(() => {
    fetchData();
  }, [symbol]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [res7d, resHist] = await Promise.all([
        axios.get(`/api/forecast/7days/${symbol}`),
        axios.get(`/api/forecast/history/${symbol}`, { params: { days: 7 } }),
      ]);
      if (res7d.data.status === 'success') setData7d(res7d.data);
      else setData7d(null);
      if (resHist.data.status === 'success') setHistoryData(resHist.data);
      else setHistoryData(null);
    } catch {
      setData7d(null);
      setHistoryData(null);
    }
    setLoading(false);
  };

  if (loading) return <Spin tip="模型预测中..." style={{ display: 'block', margin: '60px auto' }} />;
  if (!data7d) return <Card style={{ textAlign: 'center', padding: 40, background: CARD_BG, border: `1px solid ${CARD_BORDER}` }}><p style={{ color: TEXT_DIM }}>暂无预测数据</p></Card>;

  // 预测概率图
  const probData = {
    labels: data7d.predictions.map(p => `第${p.day}天`),
    datasets: [
      { label: '上涨概率(%)', data: data7d.predictions.map(p => p.upProb), borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.15)', fill: true, pointRadius: 4, borderWidth: 2 },
      { label: '下跌概率(%)', data: data7d.predictions.map(p => p.downProb), borderColor: '#f85149', backgroundColor: 'rgba(248,81,73,0.15)', fill: true, pointRadius: 4, borderWidth: 2 },
      { label: '50%分界', data: data7d.predictions.map(() => 50), borderColor: GOLD, borderWidth: 1, borderDash: [3, 3], pointRadius: 0, fill: false },
    ],
  };
  const probOpts = {
    responsive: true,
    plugins: { title: { display: true, text: `${data7d.stockName} 7天预测概率`, color: TEXT_LIGHT, font: { size: 14 } }, legend: { labels: { color: TEXT_DIM } } },
    scales: { x: { ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } }, y: { min: 0, max: 100, ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } } },
  };

  // 模拟价格图
  const priceData = {
    labels: ['当前', ...data7d.predictions.map(p => `第${p.day}天`)],
    datasets: [
      { label: '预测价格', data: [data7d.currentPrice, ...data7d.predictions.map(p => p.simPrice)], borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.08)', fill: true, pointRadius: 4, borderWidth: 2 },
      { label: '上限', data: [data7d.currentPrice, ...data7d.predictions.map(p => p.priceHigh)], borderColor: 'rgba(63,185,80,0.5)', borderWidth: 1, borderDash: [2, 2], pointRadius: 2, fill: false },
      { label: '下限', data: [data7d.currentPrice, ...data7d.predictions.map(p => p.priceLow)], borderColor: 'rgba(248,81,73,0.5)', borderWidth: 1, borderDash: [2, 2], pointRadius: 2, fill: false },
    ],
  };
  const priceOpts = {
    responsive: true,
    plugins: { title: { display: true, text: `${data7d.stockName} 预测价格走势`, color: TEXT_LIGHT, font: { size: 14 } }, legend: { labels: { color: TEXT_DIM } } },
    scales: { x: { ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } }, y: { ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } } },
  };

  // 预测明细表格
  const predColumns: TableColumnsType<Prediction7Day> = [
    { title: '天数', dataIndex: 'day', width: 60, render: (d: number) => `第${d}天` },
    { title: '日期', dataIndex: 'date', width: 100, render: (d: string) => d.slice(5) },
    { title: '方向', dataIndex: 'direction', width: 80, render: (d: string) => {
      if (d === 'up') return <Tag color="green"><RiseOutlined /> 看涨</Tag>;
      if (d === 'down') return <Tag color="red"><FallOutlined /> 看跌</Tag>;
      return <Tag>中性</Tag>;
    }},
    { title: '上涨概率', dataIndex: 'upProb', width: 100, render: (p: number) => <Progress percent={p} size="small" strokeColor={p >= 55 ? '#3fb950' : '#f85149'} format={(pct?: number) => `${pct ?? 0}%`} /> },
    { title: '预测价', dataIndex: 'simPrice', width: 80, render: (p: number) => `¥${p.toFixed(2)}` },
    { title: '价格区间', width: 120, render: (_: any, r: Prediction7Day) => `¥${r.priceLow.toFixed(2)} ~ ¥${r.priceHigh.toFixed(2)}` },
  ];

  // 历史明细表格
  const histColumns = [
    { title: '日期', dataIndex: 'date', width: 100, render: (d: string) => d.slice(5) },
    { title: '预测次数', dataIndex: 'predCount', width: 80 },
    { title: '正确次数', dataIndex: 'correctCount', width: 80 },
    { title: '准确率', dataIndex: 'accuracy', width: 100, render: (a: number) => <Tag color={a >= 55 ? 'green' : 'red'}>{a}%</Tag> },
    { title: '平均上涨概率', dataIndex: 'avgUpProb', width: 100, render: (p: number) => `${p}%` },
  ];

  // 历史柱状图
  const historyBarData = historyData ? {
    labels: historyData.dailyRecords.map(r => r.date.slice(5)),
    datasets: [{ label: '准确率(%)', data: historyData.dailyRecords.map(r => r.accuracy), backgroundColor: historyData.dailyRecords.map(r => r.accuracy >= 55 ? 'rgba(63,185,80,0.6)' : 'rgba(248,81,73,0.6)') }],
  } : null;

  const historyBarOpts = {
    responsive: true,
    plugins: { title: { display: true, text: '最近7天逐日准确率', color: TEXT_LIGHT, font: { size: 14 } }, legend: { labels: { color: TEXT_DIM } } },
    scales: { x: { ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } }, y: { min: 0, max: 100, ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } } },
  };

  const trend = data7d.summary.trendDirection;
  const trendTag = trend === 'up' ? <Tag color="green" style={{ fontSize: 14 }}><RiseOutlined /> 看涨趋势</Tag> : trend === 'down' ? <Tag color="red" style={{ fontSize: 14 }}><FallOutlined /> 看跌趋势</Tag> : <Tag style={{ fontSize: 14 }}>中性趋势</Tag>;

  return (
    <div>
      {/* 核心指标 */}
      <Card style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}`, marginBottom: 16 }} styles={{ body: { padding: 14 } }}>
        <Row gutter={16}>
          <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>当前价</span>} value={data7d.currentPrice} prefix="¥" precision={2} valueStyle={{ color: TEXT_LIGHT, fontSize: 22 }} /></Col>
          <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>趋势方向</span>} valueStyle={{ fontSize: 18 }} valueRender={() => trendTag} /></Col>
          <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>预测终价</span>} value={data7d.summary.simFinalPrice} prefix="¥" precision={2} valueStyle={{ color: TEXT_LIGHT, fontSize: 22 }} /></Col>
          <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>预测收益率</span>} value={data7d.summary.simReturn} suffix="%" precision={2} valueStyle={{ color: data7d.summary.simReturn >= 0 ? '#3fb950' : '#f85149', fontSize: 22 }} prefix={data7d.summary.simReturn >= 0 ? <RiseOutlined /> : <FallOutlined />} /></Col>
          <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>平均上涨概率</span>} value={data7d.summary.avgUpProb} suffix="%" precision={1} valueStyle={{ color: data7d.summary.avgUpProb >= 55 ? '#3fb950' : '#f85149', fontSize: 22 }} /></Col>
          {historyData && <>
            <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>历史准确率</span>} value={historyData.overallAccuracy} suffix="%" precision={1} valueStyle={{ color: historyData.overallAccuracy >= 55 ? '#3fb950' : '#f85149', fontSize: 22 }} prefix={historyData.overallAccuracy >= 55 ? <CheckCircleOutlined /> : <CloseCircleOutlined />} /></Col>
            <Col span={3}><Statistic title={<span style={{ color: TEXT_DIM }}>总预测次数</span>} value={historyData.totalPredictions} valueStyle={{ color: TEXT_LIGHT, fontSize: 22 }} /></Col>
          </>}
        </Row>
      </Card>

      {/* 概率图 + 价格图 */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          <Card style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 12 } }}>
            <Line data={probData} options={probOpts} plugins={[darkChartPlugin]} />
          </Card>
        </Col>
        <Col span={12}>
          <Card style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 12 } }}>
            <Line data={priceData} options={priceOpts} plugins={[darkChartPlugin]} />
          </Card>
        </Col>
      </Row>

      {/* 预测明细 */}
      <Card style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}`, marginBottom: 16 }} styles={{ body: { padding: 12 } }}>
        <Table columns={predColumns} dataSource={data7d.predictions} rowKey="day" pagination={false} size="small" />
      </Card>

      {/* 预测历史 */}
      {historyData && (
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={12}>
            <Card style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 12 } }}>
              {historyBarData && <Bar data={historyBarData} options={historyBarOpts} plugins={[darkChartPlugin]} />}
            </Card>
          </Col>
          <Col span={12}>
            <Card style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 12 } }}>
              <Table columns={histColumns} dataSource={historyData.dailyRecords} rowKey="date" pagination={false} size="small" />
            </Card>
          </Col>
        </Row>
      )}
    </div>
  );
}