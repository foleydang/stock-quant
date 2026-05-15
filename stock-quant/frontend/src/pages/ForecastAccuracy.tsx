import { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Button, Select, Card, Statistic, Row, Col, Tag, Spin, message, Tabs, Slider, Space } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined, RiseOutlined, FallOutlined, AimOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import axios from 'axios';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

const stockList = [
  { value: '300124.SZ', label: '汇川技术' },
  { value: '600048.SH', label: '保利发展' },
  { value: '300015.SZ', label: '爱尔眼科' },
  { value: '600519.SH', label: '贵州茅台' },
  { value: '000333.SZ', label: '美的集团' },
  { value: '600036.SH', label: '招商银行' },
  { value: '002415.SZ', label: '海康威视' },
  { value: '300750.SZ', label: '宁德时代' },
  { value: '000858.SZ', label: '五粮液' },
  { value: '600276.SH', label: '恒瑞医药' },
  { value: '601318.SH', label: '中国平安' },
  { value: '000001.SZ', label: '平安银行' },
];

interface DailyItem {
  date: string;
  actualClose: number;
  predictedClose: number;
  avgProb: number;
  directionAccuracy: number;
}

interface ForecastData {
  symbol: string;
  stockName: string;
  summary: {
    totalBars: number;
    overallAccuracy: number;
    upPrecision: number;
    downPrecision: number;
    avgUpProb: number;
    finalDeviation: number;
    finalActualPrice: number;
    finalPredictedPrice: number;
    months: number;
    avgUpChange: number;
    avgDownChange: number;
  };
  dailyComparison: DailyItem[];
  rawPredictions: {
    dates: string[];
    probs: number[];
    actualDirs: number[];
    actualPrices: number[];
    predictedPrices: number[];
  };
}

const chartPosition = 'top' as const;

// 金融深色风格主题色
const CARD_BG = '#242830';
const CARD_BORDER = '#3a3f4a';
const GOLD = '#e2b04a';
const GOLD_DIM = 'rgba(226,176,74,0.15)';
const TEXT_DIM = 'rgba(255,255,255,0.5)';
const TEXT_LIGHT = 'rgba(255,255,255,0.85)';

const darkCardStyle: React.CSSProperties = {
  background: CARD_BG,
  border: `1px solid ${CARD_BORDER}`,
  borderRadius: 8,
};



export default function ForecastAccuracy() {
  const [symbol, setSymbol] = useState('300124.SZ');
  const [months, setMonths] = useState(1);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ForecastData | null>(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`/api/forecast/accuracy/${symbol}`, {
        params: { months },
      });
      if (res.data.status === 'success') {
        setData(res.data as ForecastData);
        message.success(`${res.data.stockName} 预测验证完成`);
      } else {
        message.error(res.data.message || '加载失败');
        setData(null);
      }
    } catch (e: any) {
      message.error('请求失败：' + (e.response?.data?.message || e.message));
      setData(null);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
  }, [symbol, months]);

  // 日线对比图
  const dailyChartData = data?.dailyComparison ? {
    labels: data.dailyComparison.map((d: DailyItem) => d.date.slice(5)),
    datasets: [
      {
        label: '真实价格',
        data: data.dailyComparison.map((d: DailyItem) => d.actualClose),
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.08)',
        fill: true,
        pointRadius: 1.5,
        pointBackgroundColor: '#58a6ff',
        borderWidth: 2,
      },
      {
        label: '预测价格',
        data: data.dailyComparison.map((d: DailyItem) => d.predictedClose),
        borderColor: '#f0883e',
        backgroundColor: 'rgba(240,136,62,0.08)',
        fill: true,
        pointRadius: 1.5,
        pointBackgroundColor: '#f0883e',
        borderWidth: 2,
        borderDash: [4, 2],
      },
    ],
  } : null;

  // 预测概率分布图
  const probChartData = data?.rawPredictions ? {
    labels: data.rawPredictions.dates.map((d: string) => {
      const dt = new Date(d);
      return `${dt.getMonth()+1}/${dt.getDate()} ${dt.getHours()}:${String(dt.getMinutes()).padStart(2,'0')}`;
    }),
    datasets: [
      {
        label: '上涨概率',
        data: data.rawPredictions.probs.map((p: number) => p * 100),
        borderColor: '#3fb950',
        backgroundColor: 'rgba(63,185,80,0.15)',
        fill: true,
        pointRadius: 2,
        borderWidth: 1.5,
      },
      {
        label: '50%阈值',
        data: data.rawPredictions.probs.map(() => 50),
        borderColor: GOLD,
        borderWidth: 1,
        borderDash: [3, 3],
        pointRadius: 0,
        fill: false,
      },
    ],
  } : null;

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

  const dailyChartOptions = {
    responsive: true,
    plugins: {
      title: { display: true, text: `预测 vs 真实 — ${data?.stockName || symbol}`, color: TEXT_LIGHT, font: { size: 14 } },
      legend: { position: chartPosition, labels: { color: TEXT_DIM } },
    },
    scales: {
      x: { ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } },
      y: { title: { display: true, text: '价格 (¥)', color: TEXT_DIM }, ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } },
    },
  };

  const probChartOptions = {
    responsive: true,
    plugins: {
      title: { display: true, text: '上涨概率分布 (最近40条K线)', color: TEXT_LIGHT, font: { size: 14 } },
      legend: { position: chartPosition, labels: { color: TEXT_DIM } },
    },
    scales: {
      x: { ticks: { color: TEXT_DIM, maxTicksLimit: 15 }, grid: { color: CARD_BORDER } },
      y: { title: { display: true, text: '概率 (%)', color: TEXT_DIM }, min: 0, max: 100, ticks: { color: TEXT_DIM }, grid: { color: CARD_BORDER } },
    },
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(180deg, #1e2229 0%, #1a1e25 100%)' }}>
      {/* 导航栏 */}
      <div style={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)', padding: '14px 32px', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center', boxShadow: '0 4px 20px rgba(0,0,0,0.3)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: 1 }}>
            <AimOutlined style={{ marginRight: 10, color: GOLD }} />
            LGBM 预测准确性验证
          </h2>
        </div>
        <Space>
          <Link to="/" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none', padding: '6px 14px', borderRadius: 6, fontSize: 13, border: '1px solid rgba(255,255,255,0.15)' }}>
            主页
          </Link>
          <Link to="/trade" style={{ color: 'rgba(255,255,255,0.7)', textDecoration: 'none', padding: '6px 14px', borderRadius: 6, fontSize: 13, border: '1px solid rgba(255,255,255,0.15)' }}>
            交易记录
          </Link>
        </Space>
      </div>

      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 24px 48px' }}>
        {/* 控制栏 - 紧凑 */}
        <Card style={{ ...darkCardStyle, marginBottom: 20, padding: '4px 16px' }} styles={{ body: { padding: '12px 16px' } }}>
          <Row gutter={24} align="middle">
            <Col span={6}>
              <div style={{ color: TEXT_DIM, fontSize: 12, marginBottom: 4 }}>选择股票</div>
              <Select
                value={symbol}
                onChange={(v: string) => setSymbol(v)}
                options={stockList}
                style={{ width: '100%' }}
                size="middle"
              />
            </Col>
            <Col span={5}>
              <div style={{ color: TEXT_DIM, fontSize: 12, marginBottom: 4 }}>验证周期</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Slider min={1} max={6} value={months} onChange={(v: number) => setMonths(v)} style={{ flex: 1, margin: 0 }} />
                <Tag style={{ background: GOLD_DIM, border: `1px solid ${GOLD}`, color: GOLD, margin: 0 }}>{months}月</Tag>
              </div>
            </Col>
            <Col span={3}>
              <Button type="primary" onClick={fetchData} loading={loading} icon={<RiseOutlined />} style={{ background: GOLD, borderColor: GOLD, marginTop: 16 }}>
                验证
              </Button>
            </Col>
            <Col span={10}>
              {data && (
                <div style={{ color: TEXT_DIM, fontSize: 12, marginTop: 16 }}>
                  {data.stockName} · {data.summary.totalBars}条K线 · 终点偏离 {data.summary.finalDeviation > 0 ? '+' : ''}{data.summary.finalDeviation}%
                </div>
              )}
            </Col>
          </Row>
        </Card>

        {loading && <Spin tip="模型验证中..." style={{ display: 'block', margin: '60px auto' }} />}

        {data && !loading && (
          <>
            {/* 核心指标 - 4列紧凑 */}
            <Row gutter={12} style={{ marginBottom: 20 }}>
              <Col span={6}>
                <Card style={darkCardStyle} styles={{ body: { padding: '14px 16px' } }}>
                  <Statistic
                    title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>方向准确率</span>}
                    value={data.summary.overallAccuracy}
                    suffix="%"
                    valueStyle={{ color: data.summary.overallAccuracy >= 55 ? '#3fb950' : '#f85149', fontSize: 28, fontWeight: 700 }}
                    prefix={data.summary.overallAccuracy >= 55 ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card style={darkCardStyle} styles={{ body: { padding: '14px 16px' } }}>
                  <Statistic
                    title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>看涨精确率</span>}
                    value={data.summary.upPrecision}
                    suffix="%"
                    valueStyle={{ color: '#58a6ff', fontSize: 28, fontWeight: 700 }}
                    prefix={<RiseOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card style={darkCardStyle} styles={{ body: { padding: '14px 16px' } }}>
                  <Statistic
                    title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>看跌精确率</span>}
                    value={data.summary.downPrecision}
                    suffix="%"
                    valueStyle={{ color: '#f85149', fontSize: 28, fontWeight: 700 }}
                    prefix={<FallOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card style={darkCardStyle} styles={{ body: { padding: '14px 16px' } }}>
                  <Statistic
                    title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>终点偏离度</span>}
                    value={data.summary.finalDeviation}
                    suffix="%"
                    valueStyle={{ color: Math.abs(data.summary.finalDeviation) < 5 ? '#3fb950' : GOLD, fontSize: 28, fontWeight: 700 }}
                  />
                </Card>
              </Col>
            </Row>

            {/* 图表 Tabs */}
            <Tabs
              items={[
                {
                  key: 'daily',
                  label: <span style={{ color: TEXT_LIGHT }}>📊 日线对比</span>,
                  children: dailyChartData ? (
                    <Card style={darkCardStyle} styles={{ body: { padding: '12px 16px' } }}>
                      <Line data={dailyChartData} options={dailyChartOptions} plugins={[darkChartPlugin]} />
                      <div style={{ color: TEXT_DIM, fontSize: 11, marginTop: 8, textAlign: 'center' }}>
                        蓝=真实 · 橙虚线=预测模拟 · 偏离越小模型越准
                      </div>
                    </Card>
                  ) : null,
                },
                {
                  key: 'prob',
                  label: <span style={{ color: TEXT_LIGHT }}>📈 概率分布</span>,
                  children: probChartData ? (
                    <Card style={darkCardStyle} styles={{ body: { padding: '12px 16px' } }}>
                      <Line data={probChartData} options={probChartOptions} plugins={[darkChartPlugin]} />
                      <div style={{ color: TEXT_DIM, fontSize: 11, marginTop: 8, textAlign: 'center' }}>
                        概率{'>'}50%=看涨 · 金线=分界线
                      </div>
                    </Card>
                  ) : null,
                },
                {
                  key: 'table',
                  label: <span style={{ color: TEXT_LIGHT }}>📋 逐日明细</span>,
                  children: (
                    <Card style={darkCardStyle} styles={{ body: { padding: '8px 12px' } }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, color: TEXT_LIGHT }}>
                        <thead>
                          <tr style={{ borderBottom: `2px solid ${GOLD}` }}>
                            <th style={{ padding: 8, textAlign: 'left' }}>日期</th>
                            <th style={{ padding: 8 }}>真实</th>
                            <th style={{ padding: 8 }}>预测</th>
                            <th style={{ padding: 8 }}>偏离</th>
                            <th style={{ padding: 8 }}>概率</th>
                            <th style={{ padding: 8 }}>准确率</th>
                          </tr>
                        </thead>
                        <tbody>
                          {data.dailyComparison.map((d: DailyItem) => {
                            const dev = ((d.predictedClose - d.actualClose) / d.actualClose * 100).toFixed(2);
                            const devNum = Math.abs(parseFloat(dev));
                            return (
                              <tr key={d.date} style={{ borderBottom: `1px solid ${CARD_BORDER}` }}>
                                <td style={{ padding: 6 }}>{d.date.slice(5)}</td>
                                <td style={{ padding: 6, textAlign: 'center' }}>¥{d.actualClose.toFixed(2)}</td>
                                <td style={{ padding: 6, textAlign: 'center' }}>¥{d.predictedClose.toFixed(2)}</td>
                                <td style={{ padding: 6, textAlign: 'center' }}>
                                  <span style={{ color: devNum < 3 ? '#3fb950' : devNum < 8 ? GOLD : '#f85149' }}>{dev}%</span>
                                </td>
                                <td style={{ padding: 6, textAlign: 'center' }}>
                                  <span style={{ color: d.avgProb >= 0.5 ? '#58a6ff' : '#f85149' }}>{(d.avgProb * 100).toFixed(0)}%</span>
                                </td>
                                <td style={{ padding: 6, textAlign: 'center' }}>
                                  <span style={{ color: d.directionAccuracy >= 55 ? '#3fb950' : '#f85149' }}>{d.directionAccuracy.toFixed(0)}%</span>
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </Card>
                  ),
                },
              ]}
            />
          </>
        )}
      </div>
    </div>
  );
}