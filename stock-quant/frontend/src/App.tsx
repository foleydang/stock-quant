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
const stockList = [
  // 港股
  { value: '9988.HK', label: '🇭🇰 阿里巴巴' },
  { value: '0700.HK', label: '🇭🇰 腾讯控股' },
  { value: '3690.HK', label: '🇭🇰 美团-W' },
  { value: '159792.SZ', label: '🇭🇰 港股通互联网ETF' },
  // A股热门 - 白酒消费
  { value: '600519.SH', label: '贵州茅台' },
  { value: '000858.SZ', label: '五粮液' },
  { value: '000568.SZ', label: '泸州老窖' },
  { value: '600809.SH', label: '山西汾酒' },
  { value: '002304.SZ', label: '洋河股份' },
  { value: '600887.SH', label: '伊利股份' },
  { value: '603288.SH', label: '海天味业' },
  { value: '605499.SH', label: '东鹏饮料' },
  // A股热门 - 科技半导体
  { value: '300750.SZ', label: '宁德时代' },
  { value: '002594.SZ', label: '比亚迪' },
  { value: '002415.SZ', label: '海康威视' },
  { value: '002371.SZ', label: '北方华创' },
  { value: '300308.SZ', label: '中际旭创' },
  { value: '002475.SZ', label: '立讯精密' },
  { value: '300015.SZ', label: '爱尔眼科' },
  { value: '300760.SZ', label: '迈瑞医疗' },
  { value: '002230.SZ', label: '科大讯飞' },
  { value: '601138.SH', label: '工业富联' },
  { value: '000725.SZ', label: '京东方A' },
  { value: '688981.SH', label: '中芯国际' },
  { value: '688041.SH', label: '海光信息' },
  { value: '688256.SH', label: '寒武纪' },
  { value: '688012.SH', label: '中微公司' },
  { value: '688111.SH', label: '金山办公' },
  { value: '688271.SH', label: '联影医疗' },
  { value: '688036.SH', label: '传音控股' },
  { value: '300124.SZ', label: '汇川技术' },
  { value: '300274.SZ', label: '阳光电源' },
  { value: '300782.SZ', label: '卓胜微' },
  { value: '300059.SZ', label: '东方财富' },
  { value: '603986.SH', label: '兆易创新' },
  { value: '603019.SH', label: '中科曙光' },
  { value: '600703.SH', label: '三安光电' },
  { value: '600745.SH', label: '闻泰科技' },
  // A股热门 - 金融银行
  { value: '601398.SH', label: '工商银行' },
  { value: '601939.SH', label: '建设银行' },
  { value: '601988.SH', label: '中国银行' },
  { value: '600036.SH', label: '招商银行' },
  { value: '601318.SH', label: '中国平安' },
  { value: '601166.SH', label: '兴业银行' },
  { value: '600030.SH', label: '中信证券' },
  { value: '601688.SH', label: '华泰证券' },
  { value: '600999.SH', label: '招商证券' },
  { value: '601995.SH', label: '中金公司' },
  // A股热门 - 新能源
  { value: '601012.SH', label: '隆基绿能' },
  { value: '600438.SH', label: '通威股份' },
  { value: '603799.SH', label: '华友钴业' },
  { value: '600905.SH', label: '三峡能源' },
  { value: '601985.SH', label: '中国核电' },
  // A股热门 - 资源能源
  { value: '601899.SH', label: '紫金矿业' },
  { value: '601088.SH', label: '中国神华' },
  { value: '600028.SH', label: '中国石化' },
  { value: '601857.SH', label: '中国石油' },
  { value: '600938.SH', label: '中国海油' },
  { value: '600900.SH', label: '长江电力' },
  { value: '603993.SH', label: '洛阳钼业' },
  // A股热门 - 地产基建
  { value: '601668.SH', label: '中国建筑' },
  { value: '601390.SH', label: '中国中铁' },
  { value: '601800.SH', label: '中国交建' },
  { value: '600048.SH', label: '保利发展' },
  { value: '601816.SH', label: '京沪高铁' },
  // A股热门 - 制造化工
  { value: '600031.SH', label: '三一重工' },
  { value: '600309.SH', label: '万华化学' },
  { value: '600346.SH', label: '恒力石化' },
  { value: '000333.SZ', label: '美的集团' },
  { value: '000651.SZ', label: '格力电器' },
  { value: '600690.SH', label: '海尔智家' },
  { value: '600585.SH', label: '海螺水泥' },
  // A股热门 - 医药
  { value: '600276.SH', label: '恒瑞医药' },
  { value: '603259.SH', label: '药明康德' },
  { value: '600436.SH', label: '片仔癀' },
  // A股热门 - 其他蓝筹
  { value: '000001.SZ', label: '平安银行' },
  { value: '000002.SZ', label: '万科A' },
  { value: '002352.SZ', label: '顺丰控股' },
  { value: '000977.SZ', label: '浪潮信息' },
  { value: '600519.SH', label: '贵州茅台' },
  { value: '600570.SH', label: '恒生电子' },
  { value: '600406.SH', label: '国电南瑞' },
  { value: '600660.SH', label: '福耀玻璃' },
  { value: '601888.SH', label: '中国中免' },
  { value: '002714.SZ', label: '牧原股份' },
  { value: '601633.SH', label: '长城汽车' },
  { value: '600150.SH', label: '中国船舶' },
  { value: '600104.SH', label: '上汽集团' },
  { value: '605117.SH', label: '德业股份' },
];

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
                            <Statistic title="交易次数" value={backtestResults.summary.tradeCount} suffix="笔" />
                          </Col>
                          <Col span={3}>
                            <Statistic title="最大回撤" value={backtestResults.summary.maxDrawdown} precision={2} suffix="%" valueStyle={{ color: backtestResults.summary.maxDrawdown > 10 ? '#ff4d4f' : '#faad14' }} />
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

                      {/* 预测概率分布 - 全宽 */}
                      <Card title="预测概率分布" size="small" style={{ marginBottom: 16, background: '#242830', border: '1px solid #3a3f4a' }} styles={{ body: { padding: 12 } }}>
                        <div style={{ height: 200 }}>{predictionChart && <Bar data={predictionChart} options={volumeOptions} />}</div>
                      </Card>

                      {/* 买入/卖出点 - 并排各占50% */}
                      <Row gutter={16} style={{ marginBottom: 16 }}>
                        <Col span={12}>
                          <Card title="买入点" size="small" style={{ background: '#1a3328', border: '1px solid #3fb950' }}>
                            <div style={{ maxHeight: 200, overflow: 'auto' }}>
                              {(backtestResults?.buyPoints || []).slice(-10).map((bp: any, i: number) => (
                                <div key={i} style={{ padding: 4, borderBottom: '1px solid #3a3f4a', fontSize: 12, color: 'rgba(255,255,255,0.85)' }}>
                                  <Tag color="blue">{bp?.date?.slice(5, 10) || ""}</Tag> ¥{bp.price} | {bp.shares}股 | {bp.up_prob}%
                                </div>
                              ))}
                            </div>
                          </Card>
                        </Col>
                        <Col span={12}>
                          <Card title="卖出点" size="small" style={{ background: '#2a2818', border: '1px solid #faad14' }}>
                            <div style={{ maxHeight: 200, overflow: 'auto' }}>
                              {(backtestResults?.sellPoints || []).slice(-10).map((sp: any, i: number) => (
                                <div key={i} style={{ padding: 4, borderBottom: '1px solid #3a3f4a', fontSize: 12, color: 'rgba(255,255,255,0.85)' }}>
                                  <Tag color="orange">{sp?.date?.slice(5, 10) || ""}</Tag> ¥{sp.price} | <span style={{ color: sp.profit_pct >= 0 ? '#52c41a' : '#ff4d4f' }}>{sp.profit_pct}%</span>
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