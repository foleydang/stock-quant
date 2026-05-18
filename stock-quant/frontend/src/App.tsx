import React, { useState, useEffect } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  Filler,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Button, Select, Table, Spin, message, Card, Statistic, Row, Col, Tag, Tabs, Progress, Descriptions, DatePicker, ConfigProvider, theme } from 'antd';
import { RiseOutlined, FallOutlined, StockOutlined, FundOutlined, SwapOutlined, AimOutlined, CalculatorOutlined } from '@ant-design/icons';
import axios from 'axios';
import dayjs from 'dayjs';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import TradeRecord from './pages/TradeRecord';
import ForecastAccuracy from './pages/ForecastAccuracy';
import Forecast7Tab from './pages/Forecast7Tab';
import StockSelection from './pages/StockSelection';
import Calculator from './pages/Calculator';

const { RangePicker } = DatePicker;

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend, Filler);

// 沪深300典型蓝筹股
import { stockList } from './constants/stocks';


interface StockData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const App: React.FC = () => {
  const [symbol, setSymbol] = useState<string>('000001.SZ');
  const [period, setPeriod] = useState<string>('daily'); // daily, weekly, monthly
  const [stockData, setStockData] = useState<StockData[]>([]);
  const [stockInfo, setStockInfo] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [backtestLoading, setBacktestLoading] = useState<boolean>(false);
  const [backtestResults, setBacktestResults] = useState<any>(null);
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs | null, dayjs.Dayjs | null] | null>(() => {
    // 默认日期区间：最近 1 个月（截止到今天）
    const end = dayjs();
    const start = end.subtract(1, 'month');
    return [start, end];
  });

  // 页面加载时获取股票数据
  useEffect(() => {
    fetchStockData(symbol, period);
  }, [symbol, period]);

  // 获取股票历史数据
  const fetchStockData = async (sym: string, per: string) => {
    setLoading(true);
    try {
      const response = await axios.get(`/api/stock/${sym}/${per}`);
      const data = response.data;

      if (data.status === 'success' && data.data) {
        const history = data.data;
        setStockData(history);

        // 计算基本信息
        const prices = history.map((d: any) => d.close);
        const latestPrice = prices[prices.length - 1];
        const firstPrice = prices[0];
        const totalReturn = ((latestPrice - firstPrice) / firstPrice * 100)?.toFixed(2) || "0.00";
        const maxPrice = Math.max(...prices);
        const minPrice = Math.min(...prices);
        const avgPrice = (prices.reduce((a: number, b: number) => a + b, 0) / prices.length)?.toFixed(2) || "0.00";

        setStockInfo({
          latestPrice: latestPrice?.toFixed(2) || "0.00",
          totalReturn: totalReturn,
          maxPrice: maxPrice?.toFixed(2) || "0.00",
          minPrice: minPrice?.toFixed(2) || "0.00",
          avgPrice: avgPrice,
          dataCount: history.length,
          period: per,
        });
      }
    } catch (error) {
      console.error('获取数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 执行个股回测
  const runBacktest = async () => {
    setBacktestLoading(true);
    try {
      // 构建API URL，添加日期参数
      let url = `/api/lgbm_backtest/${symbol}`;
      if (dateRange && dateRange[0] && dateRange[1]) {
        url += `?start_date=${dateRange[0].format('YYYY-MM-DD')}&end_date=${dateRange[1].format('YYYY-MM-DD')}`;
      }

      const response = await axios.get(url);
      const data = response.data;

      if (data.status === 'success') {
        setBacktestResults(data);
        message.success(`回测完成: ${data.summary.profitRate >= 0 ? '+' : ''}${data.summary.profitRate}%`);
      } else {
        message.error('回测失败: ' + (data.error || '未知错误'));
      }
    } catch (error) {
      message.error('无法连接到服务器');
    } finally {
      setBacktestLoading(false);
    }
  };

  // 股价走势图表
  const priceChartData = {
    labels: stockData.map((d, i) => i % 8 === 0 ? d?.date?.slice(5, 10) || "" : ''),
    datasets: [
      {
        label: '收盘价',
        data: stockData.map(d => d.close),
        borderColor: '#1890ff',
        backgroundColor: 'rgba(24, 144, 255, 0.1)',
        borderWidth: 2,
        tension: 0.3,
        fill: true,
        pointRadius: 0,
      },
      {
        label: 'MA20',
        data: stockData.map((_, i, arr) => {
          if (i < 19) return null;
          const slice = arr.slice(i - 19, i + 1);
          return slice.reduce((sum, item) => sum + item.close, 0) / 20;
        }),
        borderColor: '#faad14',
        borderWidth: 1,
        tension: 0.3,
        fill: false,
        pointRadius: 0,
      },
      {
        label: 'MA60',
        data: stockData.map((_, i, arr) => {
          if (i < 59) return null;
          const slice = arr.slice(i - 59, i + 1);
          return slice.reduce((sum, item) => sum + item.close, 0) / 60;
        }),
        borderColor: '#722ed1',
        borderWidth: 1,
        tension: 0.3,
        fill: false,
        pointRadius: 0,
      },
    ],
  };

  // 成交量图表
  const volumeChartData = {
    labels: stockData.map((d, i) => i % 16 === 0 ? d?.date?.slice(5, 10) || "" : ''),
    datasets: [
      {
        label: '成交量',
        data: stockData.map(d => d.volume / 10000),
        backgroundColor: stockData.map((d, i, arr) => {
          if (i === 0) return 'rgba(82, 196, 26, 0.6)';
          return d.close >= arr[i-1].close ? 'rgba(82, 196, 26, 0.6)' : 'rgba(255, 77, 79, 0.6)';
        }),
      },
    ],
  };

  // 回测价格图表
  const backtestPriceChart = backtestResults && backtestResults?.portfolioValues ? {
    labels: backtestResults?.portfolioValues?.filter((_: any, i: number) => i % 2 === 0).map((v: any) => v?.date?.slice(5, 10) || ""),
    datasets: [
      {
        label: '股价',
        data: backtestResults?.portfolioValues?.filter((_: any, i: number) => i % 2 === 0).map((v: any) => v.price),
        borderColor: '#1890ff',
        borderWidth: 2,
        tension: 0.3,
        fill: false,
        pointRadius: 0,
      },
      {
        label: '买入点',
        data: backtestResults?.portfolioValues?.filter((_: any, i: number) => i % 2 === 0).map((v: any) => {
          const bp = backtestResults?.buyPoints?.find((b: any) => b.date === v.date);
          return bp ? bp.price : null;
        }),
        borderColor: '#1890ff',
        backgroundColor: '#1890ff',
        pointRadius: 6,
        pointStyle: 'circle',
        showLine: false,
      },
      {
        label: '卖出点',
        data: backtestResults?.portfolioValues?.filter((_: any, i: number) => i % 2 === 0).map((v: any) => {
          const sp = backtestResults?.sellPoints?.find((s: any) => s.date === v.date);
          return sp ? sp.price : null;
        }),
        borderColor: '#fa8c16',
        backgroundColor: '#fa8c16',
        pointRadius: 6,
        pointStyle: 'rectRot',
        showLine: false,
      },
    ],
  } : null;

  // 市值曲线
  const portfolioChart = backtestResults && backtestResults?.portfolioValues ? {
    labels: backtestResults?.portfolioValues?.filter((_: any, i: number) => i % 4 === 0).map((v: any) => v?.date?.slice(5, 10) || ""),
    datasets: [
      {
        label: '策略市值',
        data: backtestResults?.portfolioValues?.filter((_: any, i: number) => i % 4 === 0).map((v: any) => v.portfolioValue),
        borderColor: '#722ed1',
        backgroundColor: 'rgba(114, 46, 209, 0.1)',
        borderWidth: 2,
        tension: 0.3,
        fill: true,
      },
      {
        label: '基准(初始资金)',
        data: backtestResults?.portfolioValues?.filter((_: any, i: number) => i % 4 === 0).map(() => 100000),
        borderColor: '#d9d9d9',
        borderWidth: 1,
        borderDash: [5, 5],
        fill: false,
        pointRadius: 0,
      },
    ],
  } : null;

  // 预测概率分布
  const predictionChart = backtestResults ? {
    labels: ['<40%', '40-50%', '50-55%', '55-65%', '>65%'],
    datasets: [{
      label: '预测分布',
      data: [
        (backtestResults?.predictions || []).filter((p: any) => p.up_prob < 40).length,
        (backtestResults?.predictions || []).filter((p: any) => p.up_prob >= 40 && p.up_prob < 50).length,
        (backtestResults?.predictions || []).filter((p: any) => p.up_prob >= 50 && p.up_prob < 55).length,
        (backtestResults?.predictions || []).filter((p: any) => p.up_prob >= 55 && p.up_prob < 65).length,
        (backtestResults?.predictions || []).filter((p: any) => p.up_prob >= 65).length,
      ],
      backgroundColor: ['#ff4d4f', '#faad14', '#d9d9d9', '#52c41a', '#1890ff'],
    }],
  } : null;

  // 超额收益图表
  // 交易表格列
  const tradeColumns = [
    { title: '时间', dataIndex: 'date', render: (d: string) => d?.slice(5, 16) || "" },
    { title: '操作', dataIndex: 'type', width: 70, render: (t: string) => <Tag color={t === 'buy' ? 'green' : 'red'}>{t === 'buy' ? '买入' : '卖出'}</Tag> },
    { title: '价格', dataIndex: 'price', width: 90, render: (p: number) => `¥${p?.toFixed(2) || "0.00"}` },
    { title: '数量', dataIndex: 'shares', width: 80, render: (s: number) => `${s}股` },
    { title: '金额', dataIndex: 'amount', width: 100, render: (a: number) => a ? `¥${(a/1000)?.toFixed(1)}k` : '--' },
    { title: '盈亏', dataIndex: 'profit', width: 100, render: (p: number) => p ? <span style={{ color: p > 0 ? '#52c41a' : '#ff4d4f', fontWeight: 'bold' }}>{p > 0 ? '+' : ''}¥{p?.toFixed(0) || "0"}</span> : '--' },
    { title: '预测', dataIndex: 'up_prob', width: 80, render: (p: number) => <Tag color={p > 55 ? 'green' : p < 45 ? 'red' : 'default'}>{p}%</Tag> },
    { title: '原因', dataIndex: 'reason' },
  ];

  // 选股结果表格列
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: true, position: 'top' as const },
      tooltip: {
        enabled: true,
        intersect: false,
        mode: 'nearest' as const,
        callbacks: {
          label: (context: any) => {
            return `${context.dataset.label}: ¥${context?.parsed?.y?.toFixed(2) || "0.00"}`;
          },
          title: (items: any) => {
            const date = items[0]?.label;
            // 如果是 30 分钟线，显示具体时间
            if (period === '30m' && date) {
              const dateObj = new Date(date);
              return `${dateObj.getMonth() + 1}/${dateObj.getDate()} ${String(dateObj.getHours()).padStart(2, '0')}:${String(dateObj.getMinutes()).padStart(2, '0')}`;
            }
            return date || '';
          }
        }
      }
    },
    scales: { y: { beginAtZero: false } },
  };

  const volumeOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { y: { beginAtZero: true } },
  };

  return (
    <ConfigProvider theme={{ algorithm: theme.darkAlgorithm }}>
    <Router>
      <Routes>
        <Route path="/trade" element={
          <TradeRecord />
        } />
        <Route path="/forecast" element={
          <ForecastAccuracy />
        } />
        <Route path="/select" element={
          <StockSelection />
        } />
        <Route path="/calculator" element={
          <Calculator />
        } />
        <Route path="/" element={
          <div style={{ minHeight: '100vh', backgroundColor: '#1e2229' }}>
            {/* 顶部标题栏 - 深色金融风格 */}
            <div style={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)', padding: '16px 32px', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center', boxShadow: '0 4px 20px rgba(0,0,0,0.3)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <div>
                <h1 style={{ margin: 0, fontSize: 24, fontWeight: 600, letterSpacing: 2 }}>
                  <StockOutlined style={{ marginRight: 12, color: '#e2b04a' }} />
                  LGBM 量化交易系统
                </h1>
                <p style={{ margin: '4px 0 0', opacity: 0.6, fontSize: 13, letterSpacing: 1 }}>
                  基于机器学习的智能选股与策略回测平台
                </p>
              </div>
              <div style={{ display: 'flex', gap: 12 }}>
                <Link to="/trade" style={{ color: 'rgba(255,255,255,0.8)', textDecoration: 'none', padding: '8px 16px', background: 'rgba(226,176,74,0.15)', borderRadius: 6, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid rgba(226,176,74,0.3)' }}>
                  <SwapOutlined />
                  交易记录
                </Link>
                <Link to="/forecast" style={{ color: 'rgba(255,255,255,0.8)', textDecoration: 'none', padding: '8px 16px', background: 'rgba(226,176,74,0.15)', borderRadius: 6, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid rgba(226,176,74,0.3)' }}>
                  <AimOutlined />
                  预测验证
                </Link>
                <Link to="/select" style={{ color: 'rgba(255,255,255,0.8)', textDecoration: 'none', padding: '8px 16px', background: 'rgba(226,176,74,0.15)', borderRadius: 6, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid rgba(226,176,74,0.3)' }}>
                  <StockOutlined />
                  智能选股
                </Link>
                <Link to="/calculator" style={{ color: 'rgba(255,255,255,0.8)', textDecoration: 'none', padding: '8px 16px', background: 'rgba(226,176,74,0.15)', borderRadius: 6, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid rgba(226,176,74,0.3)' }}>
                  <CalculatorOutlined />
                  成本计算
                </Link>
              </div>
            </div>

      <div style={{ maxWidth: 1400, margin: '0 auto', padding: 24 }}>
        {/* 股票选择与概览 */}
        <Card style={{ marginBottom: 16 }}>
          <Row gutter={16} align="middle">
            <Col span={4}>
              <Select
                value={symbol}
                onChange={(v) => { setSymbol(v); setBacktestResults(null); }}
                style={{ width: '100%' }}
                options={stockList}
                size="large"
                showSearch
                filterOption={(input, option) => (option?.label ?? '').toLowerCase().includes(input.toLowerCase())}
              />
            </Col>
            <Col span={5}>
              <Button.Group>
                <Button type={period === '30m' ? 'primary' : 'default'} onClick={() => setPeriod('30m')}>30 分钟</Button>
                <Button type={period === 'daily' ? 'primary' : 'default'} onClick={() => setPeriod('daily')}>日线</Button>
                <Button type={period === 'weekly' ? 'primary' : 'default'} onClick={() => setPeriod('weekly')}>周线</Button>
                <Button type={period === 'monthly' ? 'primary' : 'default'} onClick={() => setPeriod('monthly')}>月线</Button>
              </Button.Group>
            </Col>
            <Col span={15}>
              {stockInfo && (
                <Row gutter={16}>
                  <Col span={4}>
                    <Statistic title="最新价" value={stockInfo.latestPrice} prefix="¥" valueStyle={{ fontSize: 18, color: '#e0e0e0' }} />
                  </Col>
                  <Col span={4}>
                    <Statistic title="区间涨跌" value={stockInfo.totalReturn} suffix="%" valueStyle={{ color: parseFloat(stockInfo.totalReturn) >= 0 ? '#52c41a' : '#ff4d4f', fontSize: 18 }} prefix={parseFloat(stockInfo.totalReturn) >= 0 ? <RiseOutlined /> : <FallOutlined />} />
                  </Col>
                  <Col span={4}>
                    <Statistic title="区间最高" value={stockInfo.maxPrice} prefix="¥" valueStyle={{ fontSize: 16, color: '#e0e0e0' }} />
                  </Col>
                  <Col span={4}>
                    <Statistic title="区间最低" value={stockInfo.minPrice} prefix="¥" valueStyle={{ fontSize: 16, color: '#e0e0e0' }} />
                  </Col>
                  <Col span={4}>
                    <Statistic title="区间均价" value={stockInfo.avgPrice} prefix="¥" valueStyle={{ fontSize: 16, color: '#e0e0e0' }} />
                  </Col>
                  <Col span={4}>
                    <Statistic title="数据量" value={stockInfo.dataCount} suffix="条" valueStyle={{ fontSize: 16, color: '#e0e0e0' }} />
                  </Col>
                </Row>
              )}
            </Col>
          </Row>
        </Card>

        {/* K线走势图 */}
        <Card title={`${stockList.find(s => s.value === symbol)?.label || symbol} - ${period === '30m' ? '30 分钟线' : period === 'daily' ? '日线' : period === 'weekly' ? '周线' : '月线'}走势`} style={{ marginBottom: 16 }}>
          {loading ? <Spin /> : (
            <div style={{ height: 350, marginBottom: 16 }}>
              <Line data={priceChartData} options={chartOptions} />
            </div>
          )}
          <div style={{ height: 120 }}>
            <Bar data={volumeChartData} options={volumeOptions} />
          </div>
        </Card>

        {/* 主要功能区 - 两个Tab */}
        <Tabs
          defaultActiveKey="backtest"
          items={[
            {
              key: 'forecast7',
              label: <span><AimOutlined /> 7天预测</span>,
              children: <Forecast7Tab symbol={symbol} />,
            },
            {
              key: 'backtest',
              label: <span><FundOutlined /> 策略回测</span>,
              children: (
                <div>
                  {/* 回测控制 */}
                  <Card style={{ marginBottom: 16, background: '#242830', border: '1px solid #3a3f4a' }}>
                    <Row gutter={16}>
                      <Col span={5}>
                        <Row gutter={[0, 12]}>
                          <Col span={24}>
                            <Button type="primary" size="large" onClick={runBacktest} loading={backtestLoading} block>
                              执行回测
                            </Button>
                          </Col>
                          <Col span={24}>
                            <RangePicker
                              style={{ width: '100%' }}
                              onChange={(dates) => {
                                if (dates && dates[0] && dates[1]) {
                                  setDateRange([dates[0], dates[1]]);
                                } else {
                                  setDateRange(null);
                                }
                              }}
                              value={dateRange}
                              placeholder={['开始日期', '结束日期']}
                              allowClear
                            />
                          </Col>
                        </Row>
                      </Col>
                      <Col span={19}>
                        <Descriptions column={4} size="small" labelStyle={{ color: 'rgba(255,255,255,0.5)', background: '#242830' }} contentStyle={{ color: 'rgba(255,255,255,0.85)', background: '#242830' }}>
                          <Descriptions.Item label="建仓">首次25-30%</Descriptions.Item>
                          <Descriptions.Item label="加仓">模型看涨时20%</Descriptions.Item>
                          <Descriptions.Item label="止盈">盈利10%/强看跌5%</Descriptions.Item>
                          <Descriptions.Item label="止损">亏损10%/强看跌5%</Descriptions.Item>
                          <Descriptions.Item label="买入阈值">预测上涨 &gt; 55%</Descriptions.Item>
                          <Descriptions.Item label="卖出阈值">预测下跌 &gt; 60%</Descriptions.Item>
                          <Descriptions.Item label="最小交易">5000元</Descriptions.Item>
                          <Descriptions.Item label="策略特点">LGBM 模型预测</Descriptions.Item>
                        </Descriptions>
                      </Col>
                    </Row>
                  </Card>

                  {backtestResults && (
                    <>
                      {/* 回测核心指标 */}
                      <Card style={{ marginBottom: 16, background: '#242830', border: '1px solid #3a3f4a' }}>
                        <Row gutter={16}>
                          <Col span={3}>
                            <Statistic title="总收益率" value={backtestResults.summary.profitRate} precision={2} suffix="%" valueStyle={{ color: backtestResults.summary.profitRate >= 0 ? '#52c41a' : '#ff4d4f', fontSize: 24 }} prefix={backtestResults.summary.profitRate >= 0 ? <RiseOutlined /> : <FallOutlined />} />
                          </Col>
                          <Col span={3}>
                            <Statistic title="基准收益" value={backtestResults.summary.benchmarkReturn} precision={2} suffix="%" valueStyle={{ color: backtestResults.summary.benchmarkReturn >= 0 ? '#52c41a' : '#ff4d4f', fontSize: 24 }} prefix={backtestResults.summary.benchmarkReturn >= 0 ? <RiseOutlined /> : <FallOutlined />} />
                          </Col>
                          <Col span={3}>
                            <Statistic title="超额收益" value={backtestResults.summary.excessReturn} precision={2} suffix="%" valueStyle={{ color: backtestResults.summary.excessReturn >= 0 ? '#52c41a' : '#ff4d4f', fontSize: 24 }} prefix={backtestResults.summary.excessReturn >= 0 ? <RiseOutlined /> : <FallOutlined />} />
                          </Col>
                          <Col span={3}>
                            <Statistic title="总盈亏" value={backtestResults.summary.totalProfit} precision={0} suffix="元" valueStyle={{ color: backtestResults.summary.totalProfit >= 0 ? '#52c41a' : '#ff4d4f' }} />
                          </Col>
                          <Col span={3}>
                            <Statistic title="胜率" value={backtestResults.summary.winRate} precision={1} suffix="%" valueStyle={{ color: backtestResults.summary.winRate >= 50 ? '#52c41a' : '#ff4d4f' }} />
                          </Col>
                          <Col span={3}>
                            <Statistic title="Sharpe" value={backtestResults.summary.sharpe || 0} precision={2} valueStyle={{ color: (backtestResults.summary.sharpe || 0) >= 1 ? '#52c41a' : (backtestResults.summary.sharpe || 0) >= 0 ? '#faad14' : '#ff4d4f', fontSize: 24 }} />
                          </Col>
                          <Col span={3}>
                            <Statistic title="最大回撤" value={backtestResults.summary.maxDrawdown} precision={2} suffix="%" valueStyle={{ color: backtestResults.summary.maxDrawdown > 10 ? '#ff4d4f' : '#faad14' }} />
                          </Col>
                          <Col span={3}>
                            <Statistic title="交易" value={backtestResults.summary.tradeCount} suffix="笔" valueStyle={{ color: 'rgba(255,255,255,0.85)' }} />
                          </Col>
                        </Row>
                      </Card>

                      {/* 持仓状态 */}
                      {backtestResults.summary.holdingShares > 0 && (
                        <Card style={{ marginBottom: 16, background: '#1a3328', border: '1px solid #3fb950' }}>
                          <Row gutter={16} align="middle">
                            <Col span={4}><b style={{ color: 'rgba(255,255,255,0.85)' }}>当前持仓:</b></Col>
                            <Col span={4}><Tag color="blue" style={{ fontSize: 14 }}>{backtestResults.summary.holdingShares}股</Tag></Col>
                            <Col span={4}>成本价: ¥{backtestResults.summary.avgCost}</Col>
                            <Col span={4}>市值: ¥{backtestResults.summary.finalStockValue.toFixed(0)}</Col>
                            <Col span={4}><Progress percent={Math.min(backtestResults.summary.profitRate + 50, 100)} size="small" /></Col>
                            <Col span={4}><Button type="primary" danger size="small">建议操作</Button></Col>
                          </Row>
                        </Card>
                      )}

                      {/* 图表区域 */}
                      <Row gutter={16} style={{ marginBottom: 16 }}>
                        <Col span={12}>
                          <Card title="股价与交易点" size="small" styles={{ body: { padding: 12 } }} style={{ background: '#242830', border: '1px solid #3a3f4a' }}>
                            <div style={{ height: 280 }}>{backtestPriceChart && <Line data={backtestPriceChart} options={chartOptions} />}</div>
                          </Card>
                        </Col>
                        <Col span={12}>
                          <Card title="市值曲线" size="small" styles={{ body: { padding: 12 } }} style={{ background: '#242830', border: '1px solid #3a3f4a' }}>
                            <div style={{ height: 280 }}>{portfolioChart && <Line data={portfolioChart} options={chartOptions} />}</div>
                          </Card>
                        </Col>
                      </Row>

                      {/* 预测概率分布 + 每笔收益分布 */}
                      <Row gutter={16} style={{ marginBottom: 16 }}>
                        <Col span={6}>
                          <Card title="预测概率分布" size="small" style={{ background: '#242830', border: '1px solid #3a3f4a' }}>
                            <div style={{ height: 200 }}>{predictionChart && <Bar data={predictionChart} options={volumeOptions} />}</div>
                          </Card>
                        </Col>
                        <Col span={6}>
                          <Card title="每笔交易盈亏" size="small" style={{ background: '#242830', border: '1px solid #3a3f4a' }}>
                            <div style={{ height: 200 }}>{backtestResults.trades && <Bar data={{
                              labels: backtestResults.trades.filter((t: any) => t.type === 'sell').slice(-20).map((t: any) => t.time?.slice(5, 16) || ''),
                              datasets: [{
                                label: '盈亏(元)',
                                data: backtestResults.trades.filter((t: any) => t.type === 'sell').slice(-20).map((t: any) => t.profit || 0),
                                backgroundColor: backtestResults.trades.filter((t: any) => t.type === 'sell').slice(-20).map((t: any) => (t.profit || 0) >= 0 ? 'rgba(63,185,80,0.6)' : 'rgba(248,81,73,0.6)'),
                              }],
                            }} options={{ responsive: true, maintainAspectRatio: false, plugins: { title: { display: false }, legend: { labels: { color: 'rgba(255,255,255,0.5)' } } }, scales: { x: { ticks: { color: 'rgba(255,255,255,0.4)', maxRotation: 45 }, grid: { color: '#3a3f4a' } }, y: { ticks: { color: 'rgba(255,255,255,0.4)' }, grid: { color: '#3a3f4a' } } } }} />}</div>
                          </Card>
                        </Col>
                        <Col span={6}>
                          <Card title="买入点" size="small" style={{ background: '#1a3328', border: '1px solid #3fb950' }}>
                            <div style={{ maxHeight: 200, overflow: 'auto' }}>
                              {(backtestResults?.trades || []).filter((t: any) => t.type === 'buy').slice(-10).map((t: any, i: number) => (
                                <div key={i} style={{ padding: 4, borderBottom: '1px solid #3a3f4a', fontSize: 12, color: 'rgba(255,255,255,0.85)' }}>
                                  <Tag color="blue">{t.time?.slice(5, 10) || ''}</Tag> ¥{t.price?.toFixed(2)} | {t.shares}股 | <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.6)' }}>{t.reason?.slice(0, 20) || ''}</span>
                                </div>
                              ))}
                            </div>
                          </Card>
                        </Col>
                        <Col span={6}>
                          <Card title="卖出点" size="small" style={{ background: '#2a2818', border: '1px solid #faad14' }}>
                            <div style={{ maxHeight: 200, overflow: 'auto' }}>
                              {(backtestResults?.trades || []).filter((t: any) => t.type === 'sell').slice(-10).map((t: any, i: number) => (
                                <div key={i} style={{ padding: 4, borderBottom: '1px solid #3a3f4a', fontSize: 12, color: 'rgba(255,255,255,0.85)' }}>
                                  <Tag color="orange">{t.time?.slice(5, 10) || ''}</Tag> ¥{t.price?.toFixed(2)} | {t.shares}股 | <span style={{ color: (t.profit || 0) >= 0 ? '#52c41a' : '#ff4d4f' }}>{(t.profit || 0) >= 0 ? '+' : ''}¥{(t.profit || 0).toFixed(0)}</span>
                                </div>
                              ))}
                            </div>
                          </Card>
                        </Col>
                      </Row>

                      {/* 交易记录 */}
                      <Card title="交易记录" style={{ background: '#242830', border: '1px solid #3a3f4a' }}>
                        <Table columns={tradeColumns} dataSource={backtestResults.trades} rowKey="date" pagination={{ pageSize: 15 }} size="small" locale={{ emptyText: <span style={{ color: 'rgba(255,255,255,0.4)' }}>暂无交易记录，点击上方「执行回测」开始</span> }}
                          summary={(pageData) => {
                            const sells = pageData.filter((t: any) => t.type === 'sell' && t.profit);
                            const totalProfit = sells.reduce((sum: number, t: any) => sum + (t.profit || 0), 0);
                            return (
                              <Table.Summary.Row>
                                <Table.Summary.Cell index={0} colSpan={5}><b>卖出盈亏合计</b></Table.Summary.Cell>
                                <Table.Summary.Cell index={1}><b style={{ color: totalProfit >= 0 ? '#52c41a' : '#ff4d4f' }}>¥{totalProfit.toFixed(0)}</b></Table.Summary.Cell>
                                <Table.Summary.Cell index={2} colSpan={3}></Table.Summary.Cell>
                              </Table.Summary.Row>
                            );
                          }}
                        />
                      </Card>
                    </>
                  )}
                </div>
              ),
            },
          ]}
        />
      </div>

      {/* 底部 */}
      <div style={{ textAlign: 'center', padding: 24, color: '#999', fontSize: 12 }}>
        LGBM量化交易系统 | 数据仅供参考，不构成投资建议
      </div>
    </div>
        } />
      </Routes>
    </Router>
    </ConfigProvider>
  );
};

export default App;