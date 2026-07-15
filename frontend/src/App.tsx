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
import { Button, Select, Spin, Card, Statistic, Row, Col, Tabs, ConfigProvider, theme } from 'antd';
import { RiseOutlined, FallOutlined, StockOutlined, FundOutlined, SwapOutlined, AimOutlined, CalculatorOutlined } from '@ant-design/icons';
import axios from 'axios';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import TradeRecord from './pages/TradeRecord';
import ForecastAccuracy from './pages/ForecastAccuracy';
import Forecast7Tab from './pages/Forecast7Tab';
import PortfolioBacktest from './pages/PortfolioBacktest';
import DailySignals from './pages/DailySignals';
import StockSelection from './pages/StockSelection';
import Calculator from './pages/Calculator';
import PositionManager from './pages/PositionManager';
import PaperTrading from './pages/PaperTrading';

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
  ma20?: number | null;
  ma60?: number | null;
}

const App: React.FC = () => {
  const [symbol, setSymbol] = useState<string>('000001.SZ');
  const [period, setPeriod] = useState<string>('daily'); // daily, weekly, monthly
  const [stockData, setStockData] = useState<StockData[]>([]);
  const [stockInfo, setStockInfo] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

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

  // 横轴标签: 日/周线带年份(YYYY-MM-DD), 月线 YYYY-MM, 30分钟线带具体时间(MM-DD HH:MM)
  const fmtAxis = (dateStr?: string): string => {
    if (!dateStr) return '';
    if (period === '30m') return dateStr.slice(5, 16);
    if (period === 'monthly') return dateStr.slice(0, 7);
    return dateStr.slice(0, 10);
  };

  // 股价走势图表
  const priceChartData = {
    labels: stockData.map((d, i) => i % 8 === 0 ? fmtAxis(d?.date) : ''),
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
        data: stockData.map(d => d.ma20 ?? null),
        borderColor: '#faad14',
        borderWidth: 1,
        tension: 0.3,
        fill: false,
        pointRadius: 0,
      },
      {
        label: 'MA60',
        data: stockData.map(d => d.ma60 ?? null),
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
    labels: stockData.map((d, i) => i % 16 === 0 ? fmtAxis(d?.date) : ''),
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
            // 用 dataIndex 取完整日期(标签是抽样稀疏的, 直接读会缺失)
            const idx = items?.[0]?.dataIndex;
            const full = idx != null ? stockData[idx]?.date : '';
            if (!full) return '';
            // 30分钟线显示到分钟, 其余显示到日
            return period === '30m' ? full.slice(0, 16) : full.slice(0, 10);
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
        <Route path="/signals" element={
          <DailySignals />
        } />
        <Route path="/positions" element={
          <PositionManager />
        } />
        <Route path="/paper" element={
          <PaperTrading />
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
                <Link to="/signals" style={{ color: 'rgba(255,255,255,0.8)', textDecoration: 'none', padding: '8px 16px', background: 'rgba(226,176,74,0.15)', borderRadius: 6, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid rgba(226,176,74,0.3)' }}>
                  <AimOutlined />
                  交易信号
                </Link>
                <Link to="/positions" style={{ color: 'rgba(255,255,255,0.8)', textDecoration: 'none', padding: '8px 16px', background: 'rgba(226,176,74,0.15)', borderRadius: 6, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid rgba(226,176,74,0.3)' }}>
                  <StockOutlined />
                  持仓管理
                </Link>
                <Link to="/paper" style={{ color: 'rgba(255,255,255,0.8)', textDecoration: 'none', padding: '8px 16px', background: 'rgba(226,176,74,0.15)', borderRadius: 6, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid rgba(226,176,74,0.3)' }}>
                  <FundOutlined />
                  纸面交易
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
                onChange={(v) => { setSymbol(v); }}
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
              label: <span><AimOutlined /> 20日预测</span>,
              children: <Forecast7Tab symbol={symbol} />,
            },
            {
              key: 'backtest',
              label: <span><FundOutlined /> 策略回测</span>,
              children: <PortfolioBacktest />,
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