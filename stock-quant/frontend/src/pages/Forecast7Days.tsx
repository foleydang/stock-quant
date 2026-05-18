import { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Button, Select, Card, Statistic, Row, Col, Tag, Spin, message, Tabs, Table, Space, Progress, TableColumnsType } from 'antd';
import { AimOutlined, RiseOutlined, FallOutlined, LineChartOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import axios from 'axios';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler);

const stockList = [
  { value: '000001.SZ', label: '平安银行' },
  { value: '000002.SZ', label: '万科A' },
  { value: '000333.SZ', label: '美的集团' },
  { value: '000651.SZ', label: '格力电器' },
  { value: '000858.SZ', label: '五粮液' },
  { value: '002415.SZ', label: '海康威视' },
  { value: '002594.SZ', label: '比亚迪' },
  { value: '300015.SZ', label: '爱尔眼科' },
  { value: '300124.SZ', label: '汇川技术' },
  { value: '300750.SZ', label: '宁德时代' },
  { value: '600036.SH', label: '招商银行' },
  { value: '600519.SH', label: '贵州茅台' },
  { value: '601318.SH', label: '中国平安' },
];

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
  predictions: HistoryPred[];
  accuracy: number;
  avgUpProb: number;
  predCount: number;
  correctCount: number;
}

interface HistoryPred {
  time: string;
  upProb: number;
  predDirection: string;
  actualDirection: string;
  actualReturn: number;
  isCorrect: boolean;
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

export default function Forecast7Days() {
  const [symbol, setSymbol] = useState('000001.SZ');
  const [loading7d, setLoading7d] = useState(false);
  const [loadingHist, setLoadingHist] = useState(false);
  const [data7d, setData7d] = useState<Forecast7Data | null>(null);
  const [historyData, setHistoryData] = useState<HistoryData | null>(null);
  const historyDays = 7;

  const fetch7Days = async () => {
    setLoading7d(true);
    try {
      const res = await axios.get(`/api/forecast/7days/${symbol}`);
      if (res.data.status === 'success') {
        setData7d(res.data);
        message.success(`${res.data.stockName} 7天预测完成`);
      } else {
        message.error(res.data.message || '预测失败');
        setData7d(null);
      }
    } catch (e: any) {
      message.error('请求失败：' + (e.response?.data?.message || e.message));
      setData7d(null);
    }
    setLoading7d(false);
  };

  const fetchHistory = async () => {
    setLoadingHist(true);
    try {
      const res = await axios.get(`/api/forecast/history/${symbol}`, { params: { days: historyDays } });
      if (res.data.status === 'success') {
        setHistoryData(res.data);
        message.success(`${res.data.stockName} 预测历史加载完成`);
      } else {
        message.error(res.data.message || '加载失败');
        setHistoryData(null);
      }
    } catch (e: any) {
      message.error('请求失败：' + (e.response?.data?.message || e.message));
      setHistoryData(null);
    }
    setLoadingHist(false);
  };

  useEffect(() => {
    fetch7Days();
    fetchHistory();
  }, [symbol]);

  // 7天预测概率图
  const probChartData = data7d ? {
    labels: data7d.predictions.map(p => `第${p.day}天 (${p.date.slice(5)})`),
    datasets: [
      {
        label: '上涨概率(%)',
        data: data7d.predictions.map(p => p.upProb),
        borderColor: '#3fb950',
        backgroundColor: 'rgba(63,185,80,0.15)',
        fill: true,
        pointRadius: 4,
        borderWidth: 2,
      },
      {
        label: '下跌概率(%)',
        data: data7d.predictions.map(p => p.downProb),
        borderColor: '#f85149',
        backgroundColor: 'rgba(248,81,73,0.15)',
        fill: true,
        pointRadius: 4,
        borderWidth: 2,
      },
      {
        label: '50%分界线',
        data: data7d.predictions.map(() => 50),
        borderColor: GOLD,
        borderWidth: 1,
        borderDash: [3, 3],
        pointRadius: 0,
        fill: false,
      },
    ],
  } : null;

  const probChartOptions = {
    responsive: true,
    plugins: {
      title: { display: true, text: `7天预测概率分布 — ${data7d?.stockName || symbol}`, color: TEXT_LIGHT, font: { size: 14 } },
      legend: { position: 'top' as const, labels: { color: TEXT_DIM } },
    },
    scales: {
      x: { ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } },
      y: { title: { display: true, text: '概率 (%)', color: TEXT_DIM }, min: 0, max: 100, ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } },
    },
  };

  // 模拟价格走势图
  const priceChartData = data7d ? {
    labels: ['当前', ...data7d.predictions.map(p => `第${p.day}天`)],
    datasets: [
      {
        label: '预测价格',
        data: [data7d.currentPrice, ...data7d.predictions.map(p => p.simPrice)],
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.08)',
        fill: true,
        pointRadius: 4,
        borderWidth: 2,
      },
      {
        label: '价格上限',
        data: [data7d.currentPrice, ...data7d.predictions.map(p => p.priceHigh)],
        borderColor: 'rgba(63,185,80,0.5)',
        borderWidth: 1,
        borderDash: [2, 2],
        pointRadius: 2,
        fill: false,
      },
      {
        label: '价格下限',
        data: [data7d.currentPrice, ...data7d.predictions.map(p => p.priceLow)],
        borderColor: 'rgba(248,81,73,0.5)',
        borderWidth: 1,
        borderDash: [2, 2],
        pointRadius: 2,
        fill: false,
      },
    ],
  } : null;

  const priceChartOptions = {
    responsive: true,
    plugins: {
      title: { display: true, text: `预测价格走势 — ${data7d?.stockName || symbol}`, color: TEXT_LIGHT, font: { size: 14 } },
      legend: { position: 'top' as const, labels: { color: TEXT_DIM } },
    },
    scales: {
      x: { ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } },
      y: { title: { display: true, text: '价格 (¥)', color: TEXT_DIM }, ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } },
    },
  };

  // 历史准确率柱状图
  const historyBarData = historyData ? {
    labels: historyData.dailyRecords.map(r => r.date.slice(5)),
    datasets: [
      {
        label: '准确率(%)',
        data: historyData.dailyRecords.map(r => r.accuracy),
        backgroundColor: historyData.dailyRecords.map(r => r.accuracy >= 55 ? 'rgba(63,185,80,0.6)' : 'rgba(248,81,73,0.6)'),
      },
    ],
  } : null;

  const historyBarOptions = {
    responsive: true,
    plugins: {
      title: { display: true, text: `最近${historyData?.days || 7}天逐日准确率`, color: TEXT_LIGHT, font: { size: 14 } },
      legend: { position: 'top' as const, labels: { color: TEXT_DIM } },
    },
    scales: {
      x: { ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } },
      y: { title: { display: true, text: '准确率 (%)', color: TEXT_DIM }, min: 0, max: 100, ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } },
    },
  };

  // 预测明细表格列
  const predColumns: TableColumnsType<Prediction7Day> = [
    { title: '天数', dataIndex: 'day', width: 60, render: (d: number) => `第${d}天` },
    { title: '日期', dataIndex: 'date', width: 100, render: (d: string) => d.slice(5) },
    { title: '方向', dataIndex: 'direction', width: 80, render: (d: string) => {
      if (d === 'up') return <Tag color="green"><RiseOutlined /> 看涨</Tag>;
      if (d === 'down') return <Tag color="red"><FallOutlined /> 看跌</Tag>;
      return <Tag color="default">中性</Tag>;
    }},
    { title: '上涨概率', dataIndex: 'upProb', width: 80, render: (p: number) => <Progress percent={p} size="small" strokeColor={p >= 55 ? '#3fb950' : '#f85149'} format={(pct?: number | undefined) => `${pct ?? 0}%`} /> },
    { title: '预测价', dataIndex: 'simPrice', width: 80, render: (p: number) => `¥${p.toFixed(2)}` },
    { title: '价格区间', dataIndex: 'priceLow', width: 120, render: (_: any, r: Prediction7Day) => `¥${r.priceLow.toFixed(2)} ~ ¥${r.priceHigh.toFixed(2)}` },
  ];

  // 历史明细表格列
  const histColumns = [
    { title: '日期', dataIndex: 'date', width: 100, render: (d: string) => d.slice(5) },
    { title: '预测次数', dataIndex: 'predCount', width: 80 },
    { title: '正确次数', dataIndex: 'correctCount', width: 80 },
    { title: '准确率', dataIndex: 'accuracy', width: 100, render: (a: number) => <Tag color={a >= 55 ? 'green' : 'red'}>{a}%</Tag> },
    { title: '平均上涨概率', dataIndex: 'avgUpProb', width: 100, render: (p: number) => `${p}%` },
  ];

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(180deg, #1e2229 0%, #1a1e25 100%)' }}>
      {/* 导航栏 */}
      <div style={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)', padding: '14px 32px', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center', boxShadow: '0 4px 20px rgba(0,0,0,0.3)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: 1 }}>
            <LineChartOutlined style={{ marginRight: 10, color: GOLD }} />
            7天预测 & 预测追踪
          </h2>
        </div>
        <Space>
          <Link to="/" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none', padding: '6px 14px', borderRadius: 6, fontSize: 13, border: '1px solid rgba(255,255,255,0.15)' }}>
            主页
          </Link>
          <Link to="/forecast" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none', padding: '6px 14px', borderRadius: 6, fontSize: 13, border: '1px solid rgba(255,255,255,0.15)' }}>
            预测验证
          </Link>
          <Link to="/calculator" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none', padding: '6px 14px', borderRadius: 6, fontSize: 13, border: '1px solid rgba(255,255,255,0.15)' }}>
            成本计算
          </Link>
        </Space>
      </div>

      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 24px 48px' }}>
        {/* 股票选择 */}
        <Card style={{ ...darkCardStyle, marginBottom: 20 }} styles={{ body: { padding: '12px 16px' } }}>
          <Row gutter={16} align="middle">
            <Col span={5}>
              <Select
                value={symbol}
                onChange={(v: string) => setSymbol(v)}
                options={stockList}
                style={{ width: '100%' }}
                size="middle"
                showSearch
                filterOption={(input, option) => (option?.label ?? '').toLowerCase().includes(input.toLowerCase())}
              />
            </Col>
            <Col span={3}>
              <Button type="primary" onClick={fetch7Days} loading={loading7d} style={{ background: GOLD, borderColor: GOLD }}>
                7天预测
              </Button>
            </Col>
            <Col span={4}>
              <Button onClick={fetchHistory} loading={loadingHist}>
                预测历史
              </Button>
            </Col>
          </Row>
        </Card>

        {/* 7天预测结果 */}
        {data7d && (
          <>
            {/* 核心指标 */}
            <Row gutter={12} style={{ marginBottom: 20 }}>
              <Col span={6}>
                <Card style={darkCardStyle} styles={{ body: { padding: '14px 16px' } }}>
                  <Statistic
                    title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>当前价格</span>}
                    value={data7d.currentPrice}
                    prefix="¥"
                    valueStyle={{ color: TEXT_LIGHT, fontSize: 28, fontWeight: 700 }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card style={darkCardStyle} styles={{ body: { padding: '14px 16px' } }}>
                  <Statistic
                    title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>7天趋势方向</span>}
                    value={data7d.summary.trendDirection === 'up' ? '看涨' : data7d.summary.trendDirection === 'down' ? '看跌' : '中性'}
                    valueStyle={{ color: data7d.summary.trendDirection === 'up' ? '#3fb950' : data7d.summary.trendDirection === 'down' ? '#f85149' : GOLD, fontSize: 28, fontWeight: 700 }}
                    prefix={data7d.summary.trendDirection === 'up' ? <RiseOutlined /> : data7d.summary.trendDirection === 'down' ? <FallOutlined /> : <AimOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card style={darkCardStyle} styles={{ body: { padding: '14px 16px' } }}>
                  <Statistic
                    title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>预测终价</span>}
                    value={data7d.summary.simFinalPrice}
                    prefix="¥"
                    valueStyle={{ color: TEXT_LIGHT, fontSize: 28, fontWeight: 700 }}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card style={darkCardStyle} styles={{ body: { padding: '14px 16px' } }}>
                  <Statistic
                    title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>预测收益率</span>}
                    value={data7d.summary.simReturn}
                    suffix="%"
                    valueStyle={{ color: data7d.summary.simReturn >= 0 ? '#3fb950' : '#f85149', fontSize: 28, fontWeight: 700 }}
                    prefix={data7d.summary.simReturn >= 0 ? <RiseOutlined /> : <FallOutlined />}
                  />
                </Card>
              </Col>
            </Row>

            <Tabs
              items={[
                {
                  key: 'prob',
                  label: <span style={{ color: TEXT_LIGHT }}>📊 概率分布</span>,
                  children: probChartData ? (
                    <Card style={darkCardStyle} styles={{ body: { padding: '12px 16px' } }}>
                      <Line data={probChartData} options={probChartOptions} plugins={[darkChartPlugin]} />
                    </Card>
                  ) : null,
                },
                {
                  key: 'price',
                  label: <span style={{ color: TEXT_LIGHT }}>📈 价格走势</span>,
                  children: priceChartData ? (
                    <Card style={darkCardStyle} styles={{ body: { padding: '12px 16px' } }}>
                      <Line data={priceChartData} options={priceChartOptions} plugins={[darkChartPlugin]} />
                    </Card>
                  ) : null,
                },
                {
                  key: 'table',
                  label: <span style={{ color: TEXT_LIGHT }}>📋 预测明细</span>,
                  children: (
                    <Card style={darkCardStyle} styles={{ body: { padding: '8px 12px' } }}>
                      <Table
                        columns={predColumns}
                        dataSource={data7d.predictions}
                        rowKey="day"
                        pagination={false}
                        size="small"
                      />
                    </Card>
                  ),
                },
              ]}
            />
          </>
        )}

        {/* 预测历史 */}
        {historyData && (
          <>
            <div style={{ marginTop: 24 }}>
              <Card style={darkCardStyle} styles={{ body: { padding: '14px 16px' } }}>
                <Row gutter={16}>
                  <Col span={6}>
                    <Statistic
                      title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>总预测次数</span>}
                      value={historyData.totalPredictions}
                      suffix="次"
                      valueStyle={{ color: TEXT_LIGHT, fontSize: 24 }}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>整体准确率</span>}
                      value={historyData.overallAccuracy}
                      suffix="%"
                      valueStyle={{ color: historyData.overallAccuracy >= 55 ? '#3fb950' : '#f85149', fontSize: 24, fontWeight: 700 }}
                      prefix={historyData.overallAccuracy >= 55 ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>正确次数</span>}
                      value={historyData.totalCorrect}
                      suffix="次"
                      valueStyle={{ color: '#3fb950', fontSize: 24 }}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>回测天数</span>}
                      value={historyData.days}
                      suffix="天"
                      valueStyle={{ color: TEXT_LIGHT, fontSize: 24 }}
                    />
                  </Col>
                </Row>
              </Card>
            </div>

            <Row gutter={16} style={{ marginTop: 16 }}>
              <Col span={12}>
                <Card style={darkCardStyle} styles={{ body: { padding: '12px 16px' } }}>
                  {historyBarData && <Bar data={historyBarData} options={historyBarOptions} plugins={[darkChartPlugin]} />}
                </Card>
              </Col>
              <Col span={12}>
                <Card style={darkCardStyle} styles={{ body: { padding: '8px 12px' } }}>
                  <Table
                    columns={histColumns}
                    dataSource={historyData.dailyRecords}
                    rowKey="date"
                    pagination={false}
                    size="small"
                  />
                </Card>
              </Col>
            </Row>
          </>
        )}

        {loading7d && <Spin tip="模型预测中..." style={{ display: 'block', margin: '60px auto' }} />}
        {loadingHist && <Spin tip="加载预测历史..." style={{ display: 'block', margin: '60px auto' }} />}
      </div>
    </div>
  );
}