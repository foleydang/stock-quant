import { useState } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend,
} from 'chart.js';
import { Button, Card, Statistic, Row, Col, Tag, Table, Spin, message } from 'antd';
import { StockOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import axios from 'axios';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const GOLD = '#e2b04a';

export default function StockSelection() {
  const [selectLoading, setSelectLoading] = useState(false);
  const [selectedStocks, setSelectedStocks] = useState<any[]>([]);
  const [batchResults, setBatchResults] = useState<any[]>([]);

  const runStockSelection = async () => {
    setSelectLoading(true);
    setSelectedStocks([]);
    setBatchResults([]);
    try {
      const selectRes = await axios.get('/api/select');
      const selectData = selectRes.data;
      if (selectData.status === 'success' && selectData.selected_stocks) {
        setSelectedStocks(selectData.selected_stocks.slice(0, 10));
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
              });
            }
          } catch { continue; }
        }
        results.sort((a, b) => b.profitRate - a.profitRate);
        setBatchResults(results);
        message.success(`完成 ${results.length} 只股票的策略回测`);
      }
    } catch { message.error('选股失败'); }
    finally { setSelectLoading(false); }
  };

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

  const darkLegend = { labels: { color: 'rgba(255,255,255,0.7)' } };
  const darkTicks = { color: 'rgba(255,255,255,0.5)' };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(180deg, #1e2229 0%, #1a1e25 100%)' }}>
      <div style={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)', padding: '14px 32px', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: 1 }}>
          <StockOutlined style={{ marginRight: 10, color: GOLD }} /> 智能选股
        </h2>
        <Link to="/" style={{ color: 'rgba(255,255,255,0.8)', textDecoration: 'none', padding: '8px 16px', background: 'rgba(226,176,74,0.15)', borderRadius: 6, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid rgba(226,176,74,0.3)' }}>← 返回主页</Link>
      </div>

      <div style={{ maxWidth: 1200, margin: '0 auto', padding: 24 }}>
        <Card style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 16 }} styles={{ body: { padding: 16 } }}>
          <Row gutter={16} align="middle">
            <Col span={6}>
              <Button type="primary" size="large" onClick={runStockSelection} loading={selectLoading} block icon={<StockOutlined />} style={{ background: GOLD, borderColor: GOLD }}>
                执行智能选股
              </Button>
            </Col>
            <Col span={18}>
              <p style={{ margin: 0, color: 'rgba(255,255,255,0.6)' }}>
                从沪深300成分股中，基于模型预测选出预期收益最高的股票并回测验证。
                选股标准：预测上涨概率 &gt; 55%，预测收益率排名前10。
              </p>
            </Col>
          </Row>
        </Card>

        {selectedStocks.length > 0 && (
          <Card title="选股结果" style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 16 }}>
            <Row gutter={[16, 16]}>
              {selectedStocks.map((stock, i) => (
                <Col span={6} key={i}>
                  <Card size="small" hoverable style={{ background: i < 3 ? '#1a3328' : '#242830', border: `1px solid ${i < 3 ? '#3fb950' : '#3a3f4a'}` }}>
                    <Statistic title={<span style={{ fontSize: 14, color: 'rgba(255,255,255,0.85)' }}>{stock.name} ({stock.symbol})</span>} value={stock.predicted_return} precision={2} suffix="%" valueStyle={{ color: stock.predicted_return > 0 ? '#3fb950' : '#f85149', fontSize: 20 }} />
                    <div style={{ marginTop: 8, fontSize: 12, color: 'rgba(255,255,255,0.5)' }}>现价: ¥{stock.current_price?.toFixed(2)} | 排名: #{i + 1}</div>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        )}

        {batchResults.length > 0 && (
          <>
            <Card title="策略回测对比" style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 16 }}>
              <Table columns={selectColumns} dataSource={batchResults} rowKey="symbol" pagination={false} size="small" scroll={{ x: 800 }} />
            </Card>
            <Card title="收益率对比" style={{ background: '#242830', border: '1px solid #3a3f4a', marginBottom: 16 }}>
              <div style={{ height: 300 }}>
                <Bar data={{
                  labels: batchResults.map(r => r.name),
                  datasets: [
                    { label: '回测收益率(%)', data: batchResults.map(r => r.profitRate), backgroundColor: batchResults.map(r => r.profitRate >= 0 ? 'rgba(82,196,26,0.6)' : 'rgba(255,77,79,0.6)') },
                    { label: '预测收益率(%)', data: batchResults.map(r => r.predicted_return), backgroundColor: 'rgba(24,144,255,0.4)' },
                  ],
                }} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: darkLegend }, scales: { x: { ticks: darkTicks }, y: { ticks: darkTicks } } }} />
              </div>
            </Card>
            <Card title="胜率与回撤对比" style={{ background: '#242830', border: '1px solid #3a3f4a' }}>
              <div style={{ height: 250 }}>
                <Bar data={{
                  labels: batchResults.map(r => r.name),
                  datasets: [
                    { label: '胜率(%)', data: batchResults.map(r => r.winRate), backgroundColor: 'rgba(82,196,26,0.6)' },
                    { label: '最大回撤(%)', data: batchResults.map(r => -r.maxDrawdown), backgroundColor: 'rgba(255,77,79,0.6)' },
                  ],
                }} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: darkLegend }, scales: { x: { ticks: darkTicks }, y: { ticks: darkTicks } } }} />
              </div>
            </Card>
          </>
        )}

        {!selectLoading && selectedStocks.length === 0 && (
          <Card style={{ textAlign: 'center', padding: 60, background: '#242830', border: '1px solid #3a3f4a' }}>
            <StockOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
            <p style={{ marginTop: 16, color: '#999' }}>点击上方按钮开始智能选股</p>
          </Card>
        )}

        {selectLoading && <Spin tip="正在选股..." style={{ display: 'block', margin: '60px auto' }} />}
      </div>
    </div>
  );
}