import React, { useState, useEffect } from 'react';
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
} from 'chart.js';
import { Button, Select, Table, Spin, message, Card, Statistic, Row, Col, Tag, Tabs, Progress, Descriptions } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, RiseOutlined, FallOutlined, StockOutlined, FundOutlined } from '@ant-design/icons';
import axios from 'axios';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend);

// 沪深300典型蓝筹股
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
  { value: '600000.SH', label: '浦发银行' },
  { value: '600036.SH', label: '招商银行' },
  { value: '600048.SH', label: '保利发展' },
  { value: '600276.SH', label: '恒瑞医药' },
  { value: '600309.SH', label: '万华化学' },
  { value: '600519.SH', label: '贵州茅台' },
  { value: '600585.SH', label: '海螺水泥' },
  { value: '600887.SH', label: '伊利股份' },
  { value: '600900.SH', label: '长江电力' },
  { value: '601012.SH', label: '隆基绿能' },
  { value: '601166.SH', label: '兴业银行' },
  { value: '601318.SH', label: '中国平安' },
  { value: '601398.SH', label: '工商银行' },
  { value: '601888.SH', label: '中国中免' },
  { value: '601939.SH', label: '建设银行' },
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
  const [selectLoading, setSelectLoading] = useState<boolean>(false);
  const [backtestResults, setBacktestResults] = useState<any>(null);
  const [selectedStocks, setSelectedStocks] = useState<any[]>([]);
  const [batchResults, setBatchResults] = useState<any[]>([]);

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
        const totalReturn = ((latestPrice - firstPrice) / firstPrice * 100).toFixed(2);
        const maxPrice = Math.max(...prices);
        const minPrice = Math.min(...prices);
        const avgPrice = (prices.reduce((a: number, b: number) => a + b, 0) / prices.length).toFixed(2);

        setStockInfo({
          latestPrice: latestPrice.toFixed(2),
          totalReturn: totalReturn,
          maxPrice: maxPrice.toFixed(2),
          minPrice: minPrice.toFixed(2),
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
      const response = await axios.get(`/api/lgbm_backtest/${symbol}`);
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

  // 智能选股
  const runStockSelection = async () => {
    setSelectLoading(true);
    setSelectedStocks([]);
    setBatchResults([]);

    try {
      // 1. 获取选股结果
      const selectRes = await axios.get('/api/select');
      const selectData = selectRes.data;

      if (selectData.status === 'success' && selectData.selected_stocks) {
        setSelectedStocks(selectData.selected_stocks.slice(0, 10));

        // 2. 对每只股票进行回测
        const results = [];
        for (const stock of selectData.selected_stocks.slice(0, 10)) {
          try {
            const backtestRes = await axios.get(`/api/lgbm_backtest/${stock.symbol}`);
            if (backtestRes.data.status === 'success') {
              results.push({
                symbol: stock.symbol,
                name: stock.name,
                current_price: stock.current_price,
                predicted_return: stock.predicted_return,
                ...backtestRes.data.summary,
                trades: backtestRes.data.trades,
                buyPoints: backtestRes.data.buyPoints,
                sellPoints: backtestRes.data.sellPoints,
              });
            }
          } catch {
            continue;
          }
        }

        // 按收益率排序
        results.sort((a, b) => b.profitRate - a.profitRate);
        setBatchResults(results);

        message.success(`完成 ${results.length} 只股票的策略回测`);
      }
    } catch (error) {
      message.error('选股失败');
    } finally {
      setSelectLoading(false);
    }
  };

  // 股价走势图表
  const priceChartData = {
    labels: stockData.map((d, i) => i % 8 === 0 ? d.date.slice(5, 10) : ''),
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
        data: stockData.map((d, i, arr) => {
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
        data: stockData.map((d, i, arr) => {
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
    labels: stockData.map((d, i) => i % 16 === 0 ? d.date.slice(5, 10) : ''),
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
  const backtestPriceChart = backtestResults ? {
    labels: backtestResults.portfolioValues.filter((_: any, i: number) => i % 2 === 0).map((v: any) => v.date.slice(5, 10)),
    datasets: [
      {
        label: '股价',
        data: backtestResults.portfolioValues.filter((_: any, i: number) => i % 2 === 0).map((v: any) => v.price),
        borderColor: '#1890ff',
        borderWidth: 2,
        tension: 0.3,
        fill: false,
        pointRadius: 0,
      },
      {
        label: '买入点',
        data: backtestResults.portfolioValues.filter((_: any, i: number) => i % 2 === 0).map((v: any) => {
          const bp = backtestResults.buyPoints.find((b: any) => b.date === v.date);
          return bp ? bp.price : null;
        }),
        borderColor: '#52c41a',
        backgroundColor: '#52c41a',
        pointRadius: 6,
        pointStyle: 'triangle',
        showLine: false,
      },
      {
        label: '卖出点',
        data: backtestResults.portfolioValues.filter((_: any, i: number) => i % 2 === 0).map((v: any) => {
          const sp = backtestResults.sellPoints.find((s: any) => s.date === v.date);
          return sp ? sp.price : null;
        }),
        borderColor: '#ff4d4f',
        backgroundColor: '#ff4d4f',
        pointRadius: 6,
        pointStyle: 'rectRot',
        showLine: false,
      },
    ],
  } : null;

  // 市值曲线
  const portfolioChart = backtestResults ? {
    labels: backtestResults.portfolioValues.filter((_: any, i: number) => i % 4 === 0).map((v: any) => v.date.slice(5, 10)),
    datasets: [
      {
        label: '总市值',
        data: backtestResults.portfolioValues.filter((_: any, i: number) => i % 4 === 0).map((v: any) => v.portfolioValue),
        borderColor: '#722ed1',
        backgroundColor: 'rgba(114, 46, 209, 0.1)',
        borderWidth: 2,
        tension: 0.3,
        fill: true,
      },
      {
        label: '基准(初始资金)',
        data: backtestResults.portfolioValues.filter((_: any, i: number) => i % 4 === 0).map(() => 100000),
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
        backtestResults.predictions.filter((p: any) => p.up_prob < 40).length,
        backtestResults.predictions.filter((p: any) => p.up_prob >= 40 && p.up_prob < 50).length,
        backtestResults.predictions.filter((p: any) => p.up_prob >= 50 && p.up_prob < 55).length,
        backtestResults.predictions.filter((p: any) => p.up_prob >= 55 && p.up_prob < 65).length,
        backtestResults.predictions.filter((p: any) => p.up_prob >= 65).length,
      ],
      backgroundColor: ['#ff4d4f', '#faad14', '#d9d9d9', '#52c41a', '#1890ff'],
    }],
  } : null;

  // 交易表格列
  const tradeColumns = [
    { title: '时间', dataIndex: 'date', width: 100, render: (d: string) => d.slice(5, 16) },
    { title: '操作', dataIndex: 'type', width: 60, render: (t: string) => <Tag color={t === 'BUY' ? 'green' : 'red'}>{t === 'BUY' ? '买入' : '卖出'}</Tag> },
    { title: '价格', dataIndex: 'price', width: 70, render: (p: number) => `¥${p.toFixed(2)}` },
    { title: '数量', dataIndex: 'shares', width: 60, render: (s: number) => `${s}股` },
    { title: '金额', dataIndex: 'amount', width: 80, render: (a: number) => `¥${(a/1000).toFixed(1)}k` },
    { title: '盈亏', dataIndex: 'profit', width: 80, render: (p: number) => p ? <span style={{ color: p > 0 ? '#52c41a' : '#ff4d4f', fontWeight: 'bold' }}>{p > 0 ? '+' : ''}¥{p.toFixed(0)}</span> : '--' },
    { title: '预测', dataIndex: 'up_prob', width: 60, render: (p: number) => <Tag color={p > 55 ? 'green' : p < 45 ? 'red' : 'default'}>{p}%</Tag> },
    { title: '原因', dataIndex: 'reason', ellipsis: true, render: (r: string) => r?.slice(0, 15) },
  ];

  // 选股结果表格列
  const selectColumns = [
    { title: '排名', dataIndex: 'rank', width: 50, render: (_: any, __: any, i: number) => i + 1 },
    { title: '股票', dataIndex: 'name', width: 100 },
    { title: '代码', dataIndex: 'symbol', width: 100 },
    { title: '现价', dataIndex: 'current_price', width: 70, render: (p: number) => `¥${p?.toFixed(2)}` },
    { title: '预测收益', dataIndex: 'predicted_return', width: 90, render: (r: number) => <Tag color={r > 0 ? 'green' : 'red'}>{r?.toFixed(2)}%</Tag> },
    { title: '回测收益', dataIndex: 'profitRate', width: 90, render: (r: number) => <span style={{ color: r >= 0 ? '#52c41a' : '#ff4d4f', fontWeight: 'bold' }}>{r >= 0 ? '+' : ''}{r?.toFixed(2)}%</span> },
    { title: '胜率', dataIndex: 'winRate', width: 70, render: (r: number) => <span style={{ color: r >= 50 ? '#52c41a' : '#ff4d4f' }}>{r?.toFixed(1)}%</span> },
    { title: '交易次数', dataIndex: 'tradeCount', width: 80 },
    { title: '最大回撤', dataIndex: 'maxDrawdown', width: 90, render: (d: number) => <span style={{ color: d > 10 ? '#ff4d4f' : '#faad14' }}>-{d?.toFixed(2)}%</span> },
  ];

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: true, position: 'top' as const } },
    scales: { y: { beginAtZero: false } },
  };

  const volumeOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { y: { beginAtZero: true } },
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f0f2f5' }}>
      {/* 顶部标题栏 */}
      <div style={{ background: 'linear-gradient(135deg, #1890ff 0%, #722ed1 100%)', padding: '20px 24px', color: 'white' }}>
        <h1 style={{ margin: 0, fontSize: 28 }}>
          <StockOutlined style={{ marginRight: 12 }} />
          LGBM 量化交易系统
        </h1>
        <p style={{ margin: '8px 0 0', opacity: 0.9 }}>
          基于机器学习的智能选股与策略回测平台
        </p>
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
            <Col span={4}>
              <Button.Group size="large">
                <Button type={period === 'daily' ? 'primary' : 'default'} onClick={() => setPeriod('daily')}>日线</Button>
                <Button type={period === 'weekly' ? 'primary' : 'default'} onClick={() => setPeriod('weekly')}>周线</Button>
                <Button type={period === 'monthly' ? 'primary' : 'default'} onClick={() => setPeriod('monthly')}>月线</Button>
              </Button.Group>
            </Col>
            <Col span={16}>
              {stockInfo && (
                <Row gutter={24}>
                  <Col span={4}>
                    <Statistic title="最新价" value={stockInfo.latestPrice} prefix="¥" />
                  </Col>
                  <Col span={4}>
                    <Statistic title="区间涨跌" value={stockInfo.totalReturn} suffix="%" valueStyle={{ color: parseFloat(stockInfo.totalReturn) >= 0 ? '#52c41a' : '#ff4d4f' }} prefix={parseFloat(stockInfo.totalReturn) >= 0 ? <RiseOutlined /> : <FallOutlined />} />
                  </Col>
                  <Col span={4}>
                    <Statistic title="区间最高" value={stockInfo.maxPrice} prefix="¥" />
                  </Col>
                  <Col span={4}>
                    <Statistic title="区间最低" value={stockInfo.minPrice} prefix="¥" />
                  </Col>
                  <Col span={4}>
                    <Statistic title="区间均价" value={stockInfo.avgPrice} prefix="¥" />
                  </Col>
                  <Col span={4}>
                    <Statistic title="数据量" value={stockInfo.dataCount} suffix="条" />
                  </Col>
                </Row>
              )}
            </Col>
          </Row>
        </Card>

        {/* K线走势图 */}
        <Card title={`${stockList.find(s => s.value === symbol)?.label || symbol} - ${period === 'daily' ? '日线' : period === 'weekly' ? '周线' : '月线'}走势`} style={{ marginBottom: 16 }}>
          {loading ? <Spin /> : (
            <>
              <div style={{ height: 350, marginBottom: 16 }}>
                <Line data={priceChartData} options={chartOptions} />
              </div>
              <div style={{ height: 120 }}>
                <Bar data={volumeChartData} options={volumeOptions} />
              </div>
            </>
          )}
        </Card>

        {/* 主要功能区 - 两个Tab */}
        <Tabs
          defaultActiveKey="backtest"
          items={[
            {
              key: 'backtest',
              label: <span><FundOutlined /> 个股策略回测</span>,
              children: (
                <div>
                  {/* 回测控制 */}
                  <Card style={{ marginBottom: 16 }}>
                    <Row gutter={16} align="middle">
                      <Col span={4}>
                        <Button type="primary" size="large" onClick={runBacktest} loading={backtestLoading} block>
                          执行回测
                        </Button>
                      </Col>
                      <Col span={20}>
                        <Descriptions column={4} size="small">
                          <Descriptions.Item label="建仓">首次25-30%</Descriptions.Item>
                          <Descriptions.Item label="加仓">模型看涨时20%</Descriptions.Item>
                          <Descriptions.Item label="止盈">盈利10%且模型看跌</Descriptions.Item>
                          <Descriptions.Item label="止损">亏损10%且模型看跌</Descriptions.Item>
                          <Descriptions.Item label="买入阈值">预测上涨 &gt; 55%</Descriptions.Item>
                          <Descriptions.Item label="卖出阈值">预测下跌 &gt; 60%</Descriptions.Item>
                          <Descriptions.Item label="最小交易">5000元</Descriptions.Item>
                          <Descriptions.Item label="策略特点">基于模型预测，无机械比例限制</Descriptions.Item>
                        </Descriptions>
                      </Col>
                    </Row>
                  </Card>

                  {backtestResults && (
                    <>
                      {/* 回测核心指标 */}
                      <Card style={{ marginBottom: 16 }}>
                        <Row gutter={16}>
                          <Col span={4}>
                            <Statistic title="总收益率" value={backtestResults.summary.profitRate} precision={2} suffix="%" valueStyle={{ color: backtestResults.summary.profitRate >= 0 ? '#52c41a' : '#ff4d4f', fontSize: 28 }} prefix={backtestResults.summary.profitRate >= 0 ? <RiseOutlined /> : <FallOutlined />} />
                          </Col>
                          <Col span={4}>
                            <Statistic title="总盈亏" value={backtestResults.summary.totalProfit} precision={0} suffix="元" valueStyle={{ color: backtestResults.summary.totalProfit >= 0 ? '#52c41a' : '#ff4d4f' }} />
                          </Col>
                          <Col span={4}>
                            <Statistic title="胜率" value={backtestResults.summary.winRate} precision={1} suffix="%" valueStyle={{ color: backtestResults.summary.winRate >= 50 ? '#52c41a' : '#ff4d4f' }} />
                          </Col>
                          <Col span={4}>
                            <Statistic title="交易次数" value={backtestResults.summary.tradeCount} suffix="笔" />
                          </Col>
                          <Col span={4}>
                            <Statistic title="最大回撤" value={backtestResults.summary.maxDrawdown} precision={2} suffix="%" valueStyle={{ color: backtestResults.summary.maxDrawdown > 10 ? '#ff4d4f' : '#faad14' }} />
                          </Col>
                          <Col span={4}>
                            <Statistic title="模型准确率" value={backtestResults.summary.modelAccuracy} precision={1} suffix="%" />
                          </Col>
                        </Row>
                      </Card>

                      {/* 持仓状态 */}
                      {backtestResults.summary.holdingShares > 0 && (
                        <Card style={{ marginBottom: 16, background: '#e6f7ff' }}>
                          <Row gutter={16} align="middle">
                            <Col span={4}><b>当前持仓:</b></Col>
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
                          <Card title="股价与交易点" size="small">
                            <div style={{ height: 280 }}>{backtestPriceChart && <Line data={backtestPriceChart} options={chartOptions} />}</div>
                          </Card>
                        </Col>
                        <Col span={12}>
                          <Card title="市值曲线" size="small">
                            <div style={{ height: 280 }}>{portfolioChart && <Line data={portfolioChart} options={chartOptions} />}</div>
                          </Card>
                        </Col>
                      </Row>

                      <Row gutter={16} style={{ marginBottom: 16 }}>
                        <Col span={12}>
                          <Card title="预测概率分布" size="small">
                            <div style={{ height: 200 }}>{predictionChart && <Bar data={predictionChart} options={volumeOptions} />}</div>
                          </Card>
                        </Col>
                        <Col span={6}>
                          <Card title="买入点" size="small" style={{ background: '#f6ffed' }}>
                            <div style={{ maxHeight: 200, overflow: 'auto' }}>
                              {backtestResults.buyPoints.slice(-10).map((bp: any, i: number) => (
                                <div key={i} style={{ padding: 4, borderBottom: '1px solid #e8e8e8', fontSize: 12 }}>
                                  <Tag color="green">{bp.date.slice(5, 10)}</Tag> ¥{bp.price} | {bp.shares}股 | {bp.up_prob}%
                                </div>
                              ))}
                            </div>
                          </Card>
                        </Col>
                        <Col span={6}>
                          <Card title="卖出点" size="small" style={{ background: '#fff2f0' }}>
                            <div style={{ maxHeight: 200, overflow: 'auto' }}>
                              {backtestResults.sellPoints.slice(-10).map((sp: any, i: number) => (
                                <div key={i} style={{ padding: 4, borderBottom: '1px solid #e8e8e8', fontSize: 12 }}>
                                  <Tag color="red">{sp.date.slice(5, 10)}</Tag> ¥{sp.price} | <span style={{ color: sp.profit_pct >= 0 ? '#52c41a' : '#ff4d4f' }}>{sp.profit_pct}%</span>
                                </div>
                              ))}
                            </div>
                          </Card>
                        </Col>
                      </Row>

                      {/* 交易记录 */}
                      <Card title="交易记录">
                        <Table columns={tradeColumns} dataSource={backtestResults.trades} rowKey="date" pagination={{ pageSize: 15 }} size="small" scroll={{ x: 700 }}
                          summary={(pageData) => {
                            const sells = pageData.filter((t: any) => t.type === 'SELL' && t.profit);
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
            {
              key: 'selection',
              label: <span><StockOutlined /> 股票池智能选股</span>,
              children: (
                <div>
                  <Card style={{ marginBottom: 16 }}>
                    <Row gutter={16} align="middle">
                      <Col span={6}>
                        <Button type="primary" size="large" onClick={runStockSelection} loading={selectLoading} block icon={<StockOutlined />}>
                          执行智能选股
                        </Button>
                      </Col>
                      <Col span={18}>
                        <p style={{ margin: 0, color: '#666' }}>
                          系统将从沪深300成分股中，基于模型预测选出预期收益最高的股票，并对每只股票进行策略回测验证。
                          选股标准：预测上涨概率 &gt; 55%，预测收益率排名前10。
                        </p>
                      </Col>
                    </Row>
                  </Card>

                  {selectedStocks.length > 0 && (
                    <Card title="选股结果" style={{ marginBottom: 16 }}>
                      <Row gutter={[16, 16]}>
                        {selectedStocks.map((stock, i) => (
                          <Col span={6} key={i}>
                            <Card size="small" hoverable style={{ background: i < 3 ? '#f6ffed' : '#fff' }}>
                              <Statistic
                                title={<span style={{ fontSize: 14 }}>{stock.name} ({stock.symbol})</span>}
                                value={stock.predicted_return}
                                precision={2}
                                suffix="%"
                                valueStyle={{ color: stock.predicted_return > 0 ? '#52c41a' : '#ff4d4f', fontSize: 20 }}
                              />
                              <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
                                现价: ¥{stock.current_price?.toFixed(2)} | 排名: #{i + 1}
                              </div>
                            </Card>
                          </Col>
                        ))}
                      </Row>
                    </Card>
                  )}

                  {batchResults.length > 0 && (
                    <>
                      <Card title="策略回测对比" style={{ marginBottom: 16 }}>
                        <Table
                          columns={selectColumns}
                          dataSource={batchResults}
                          rowKey="symbol"
                          pagination={false}
                          size="small"
                          scroll={{ x: 800 }}
                          onRow={(record) => ({
                            onClick: () => {
                              setSymbol(record.symbol);
                              // 切换到个股回测tab
                            },
                            style: { cursor: 'pointer' },
                          })}
                        />
                      </Card>

                      {/* 收益对比图 */}
                      <Card title="收益率对比" style={{ marginBottom: 16 }}>
                        <div style={{ height: 300 }}>
                          <Bar
                            data={{
                              labels: batchResults.map(r => r.name),
                              datasets: [
                                {
                                  label: '回测收益率(%)',
                                  data: batchResults.map(r => r.profitRate),
                                  backgroundColor: batchResults.map(r => r.profitRate >= 0 ? 'rgba(82, 196, 26, 0.6)' : 'rgba(255, 77, 79, 0.6)'),
                                },
                                {
                                  label: '预测收益率(%)',
                                  data: batchResults.map(r => r.predicted_return),
                                  backgroundColor: 'rgba(24, 144, 255, 0.4)',
                                },
                              ],
                            }}
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              plugins: { legend: { position: 'top' } },
                            }}
                          />
                        </div>
                      </Card>

                      {/* 胜率对比 */}
                      <Card title="胜率与回撤对比">
                        <div style={{ height: 250 }}>
                          <Bar
                            data={{
                              labels: batchResults.map(r => r.name),
                              datasets: [
                                { label: '胜率(%)', data: batchResults.map(r => r.winRate), backgroundColor: 'rgba(82, 196, 26, 0.6)' },
                                { label: '最大回撤(%)', data: batchResults.map(r => -r.maxDrawdown), backgroundColor: 'rgba(255, 77, 79, 0.6)' },
                              ],
                            }}
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              plugins: { legend: { position: 'top' } },
                            }}
                          />
                        </div>
                      </Card>
                    </>
                  )}

                  {!selectLoading && selectedStocks.length === 0 && (
                    <Card style={{ textAlign: 'center', padding: 60 }}>
                      <StockOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
                      <p style={{ marginTop: 16, color: '#999' }}>点击上方按钮开始智能选股</p>
                    </Card>
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
  );
};

export default App;