import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { Card, Form, Input, InputNumber, Select, Button, message, Table, Tag, Row, Col, AutoComplete } from 'antd';
import { SwapOutlined } from '@ant-design/icons';
import axios from 'axios';

interface Position {
  symbol: string;
  name: string;
  shares: number;
  cost: number;
  current: number;
}

interface Trade {
  symbol: string;
  name: string;
  action: string;
  shares: number;
  price: number;
  amount: number;
  reason: string;
  timestamp: string;
}

interface StockOption {
  value: string;
  label: string;
  symbol: string;
  name: string;
}

const TradeRecord: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [form] = Form.useForm();

  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(false);
  const [stockOptions, setStockOptions] = useState<StockOption[]>([]);
  const [filteredOptions, setFilteredOptions] = useState<StockOption[]>([]);

  // 加载持仓和交易记录
  useEffect(() => {
    fetchPositions();
    fetchTrades();
    fetchStockList();
  }, []);

  // 预填股票信息（从 URL 参数）
  useEffect(() => {
    const symbolFromUrl = searchParams.get('symbol');
    const nameFromUrl = searchParams.get('name');
    if (symbolFromUrl) {
      form.setFieldsValue({
        symbol: symbolFromUrl,
        stock_name: nameFromUrl || '',
      });
    }
  }, [searchParams]);

  const fetchPositions = async () => {
    try {
      const res = await axios.get('/api/positions');
      if (res.data.status === 'success') {
        setPositions(res.data.positions);
      }
    } catch (error) {
      console.error('获取持仓失败:', error);
    }
  };

  const fetchTrades = async () => {
    try {
      const res = await axios.get('/api/trades');
      if (res.data.status === 'success') {
        setTrades(res.data.trades);
      }
    } catch (error) {
      console.error('获取交易记录失败:', error);
    }
  };

  const fetchStockList = async () => {
    try {
      const res = await axios.get('/api/db/stocks');
      if (res.data.status === 'success' && res.data.stocks) {
        const options = res.data.stocks.map((stock: any) => ({
          value: `${stock.symbol} - ${stock.name}`,
          label: `${stock.symbol} - ${stock.name}`,
          symbol: stock.symbol,
          name: stock.name,
        }));
        setStockOptions(options);
        setFilteredOptions(options);
      }
    } catch (error) {
      console.error('获取股票列表失败:', error);
    }
  };

  const handleSymbolSearch = (value: string) => {
    if (!value) {
      setFilteredOptions(stockOptions);
      return;
    }

    const searchLower = value.toLowerCase();
    const filtered = stockOptions.filter((option) => {
      return option.symbol.toLowerCase().includes(searchLower) ||
             option.name.toLowerCase().includes(searchLower);
    });
    setFilteredOptions(filtered);
  };

  const handleSymbolSelect = (value: string) => {
    const selected = stockOptions.find(opt => opt.value === value);
    if (selected) {
      form.setFieldsValue({
        stock_name: selected.name,
      });
    }
  };

  const handleSubmit = async (values: any) => {
    setLoading(true);
    try {
      const res = await axios.post('/api/trade', values);
      if (res.data.status === 'success') {
        message.success(res.data.message);
        form.resetFields();
        fetchPositions();
        fetchTrades();
      } else {
        message.error(res.data.error || '交易记录失败');
      }
    } catch (error: any) {
      message.error(error.response?.data?.error || '交易记录失败');
    } finally {
      setLoading(false);
    }
  };

  // 交易记录表格列
  const tradeColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 160,
      render: (t: string) => t ? new Date(t).toLocaleString('zh-CN') : '-',
    },
    {
      title: '股票',
      key: 'stock',
      width: 150,
      render: (_: any, record: Trade) => `${record.name} (${record.symbol})`,
    },
    {
      title: '操作',
      dataIndex: 'action',
      key: 'action',
      width: 70,
      render: (action: string) => (
        <Tag color={action === 'BUY' ? 'green' : 'red'}>
          {action === 'BUY' ? '买入' : '卖出'}
        </Tag>
      ),
    },
    {
      title: '价格',
      dataIndex: 'price',
      key: 'price',
      width: 80,
      render: (p: number) => `¥${p?.toFixed(2) || 0.00}`,
    },
    {
      title: '数量',
      dataIndex: 'shares',
      key: 'shares',
      width: 80,
      render: (s: number) => `${s.toLocaleString()}股`,
    },
    {
      title: '金额',
      dataIndex: 'amount',
      key: 'amount',
      width: 100,
      render: (a: number) => `¥${a.toLocaleString()}`,
    },
    {
      title: '原因',
      dataIndex: 'reason',
      key: 'reason',
      ellipsis: true,
    },
  ];

  // 持仓表格列（移除操作列）
  const positionColumns = [
    {
      title: '股票',
      key: 'stock',
      width: 180,
      render: (_: any, record: Position) => `${record.name} (${record.symbol})`,
    },
    {
      title: '持股数量',
      dataIndex: 'shares',
      key: 'shares',
      width: 100,
      render: (s: number) => `${s.toLocaleString()}股`,
    },
    {
      title: '成本价',
      dataIndex: 'cost',
      key: 'cost',
      width: 90,
      render: (p: number) => `¥${p?.toFixed(3) || 0.000}`,
    },
    {
      title: '现价',
      dataIndex: 'current',
      key: 'current',
      width: 90,
      render: (p: number) => `¥${p?.toFixed(2) || 0.00}`,
    },
    {
      title: '盈亏',
      key: 'profit',
      width: 120,
      render: (_: any, record: Position) => {
        const profit = (record.current - record.cost) * record.shares;
        const profitPct = ((record.current - record.cost) / record.cost * 100);
        return (
          <span style={{ color: profit >= 0 ? '#52c41a' : '#ff4d4f', fontWeight: 'bold' }}>
            ¥{profit.toLocaleString(undefined, { maximumFractionDigits: 0 })} ({profitPct?.toFixed(1) || 0}%)
          </span>
        );
      },
    },
  ];

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#1e2229', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* 标题 */}
        <Card style={{ marginBottom: 16 }}>
          <Row gutter={16} align="middle">
            <Col span={12}>
              <h1 style={{ margin: 0, display: 'flex', alignItems: 'center', color: '#e0e0e0' }}>
                <SwapOutlined style={{ marginRight: 12, color: '#e2b04a' }} />
                交易记录
              </h1>
              <p style={{ margin: '8px 0 0', color: '#8899aa' }}>
                记录您的买入/卖出操作，同步持仓数据
              </p>
            </Col>
            <Col span={12} style={{ textAlign: 'right' }}>
              <Button onClick={() => navigate('/')}>返回首页</Button>
            </Col>
          </Row>
        </Card>

        {/* 当前持仓 - 放在顶部 */}
        <Card title="📊 当前持仓" style={{ marginBottom: 16 }}>
          {positions.length > 0 ? (
            <Table
              columns={positionColumns}
              dataSource={positions}
              rowKey="symbol"
              pagination={false}
              size="middle"
            />
          ) : (
            <p style={{ textAlign: 'center', color: '#8899aa', padding: 20 }}>暂无持仓</p>
          )}
        </Card>

        <Row gutter={16}>
          {/* 左侧：交易录入表单 */}
          <Col span={10}>
            <Card title="📝 录入交易">
              <Form
                form={form}
                layout="vertical"
                onFinish={handleSubmit}
                initialValues={{
                  action: 'BUY',
                  shares: 100,
                }}
              >
                <Form.Item
                  name="symbol"
                  label="股票代码"
                  rules={[{ required: true, message: '请输入或选择股票代码' }]}
                  extra="支持输入代码或名称搜索，如：000001 或 平安银行"
                >
                  <AutoComplete
                    options={filteredOptions}
                    filterOption={false}
                    onSearch={handleSymbolSearch}
                    onSelect={handleSymbolSelect}
                    placeholder="输入股票代码或名称"
                    style={{ width: '100%' }}
                    allowClear
                  >
                    {filteredOptions.map((opt) => (
                      <AutoComplete.Option key={opt.value} value={opt.value}>
                        <span style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <strong>{opt.symbol}</strong>
                          <span style={{ color: '#8899aa' }}>{opt.name}</span>
                        </span>
                      </AutoComplete.Option>
                    ))}
                  </AutoComplete>
                </Form.Item>

                <Form.Item
                  name="stock_name"
                  label="股票名称"
                  rules={[{ required: true, message: '请输入股票名称' }]}
                >
                  <Input placeholder="选择股票代码后自动填充" readOnly style={{ backgroundColor: '#2a2f38' }} />
                </Form.Item>

                <Form.Item
                  name="action"
                  label="操作类型"
                  rules={[{ required: true, message: '请选择操作类型' }]}
                >
                  <Select>
                    <Select.Option value="BUY" style={{ color: '#52c41a' }}>🟢 买入</Select.Option>
                    <Select.Option value="SELL" style={{ color: '#ff4d4f' }}>🔴 卖出</Select.Option>
                  </Select>
                </Form.Item>

                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item
                      name="shares"
                      label="数量（股）"
                      rules={[{ required: true, message: '请输入数量' }]}
                    >
                      <InputNumber
                        min={100}
                        step={100}
                        placeholder="100"
                        style={{ width: '100%' }}
                        addonAfter="股"
                      />
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item
                      name="price"
                      label="价格（元）"
                      rules={[{ required: true, message: '请输入价格' }]}
                    >
                      <InputNumber
                        min={0.01}
                        step={0.01}
                        placeholder="0.00"
                        style={{ width: '100%' }}
                        precision={2}
                        addonAfter="元"
                      />
                    </Form.Item>
                  </Col>
                </Row>

                <Form.Item
                  name="reason"
                  label="交易原因（可选）"
                >
                  <Input.TextArea
                    rows={2}
                    placeholder="如：补仓、做 T、止损等"
                  />
                </Form.Item>

                <Form.Item>
                  <Button type="primary" htmlType="submit" loading={loading} block size="large">
                    确认记录
                  </Button>
                </Form.Item>
              </Form>
            </Card>
          </Col>

          {/* 右侧：交易历史记录 */}
          <Col span={14}>
            <Card title="📜 交易历史记录">
              <Table
                columns={tradeColumns}
                dataSource={trades}
                rowKey={(record) => `${record.symbol}-${record.timestamp}`}
                pagination={{ pageSize: 20 }}
                size="middle"
                scroll={{ x: 800 }}
                summary={(pageData) => {
                  const buys = pageData.filter(t => t.action === 'BUY');
                  const sells = pageData.filter(t => t.action === 'SELL');
                  const totalBuyAmount = buys.reduce((sum, t) => sum + t.amount, 0);
                  const totalSellAmount = sells.reduce((sum, t) => sum + t.amount, 0);

                  return (
                    <>
                      <Table.Summary.Row>
                        <Table.Summary.Cell index={0} colSpan={3}>
                          <b style={{ color: '#52c41a' }}>买入合计</b>
                        </Table.Summary.Cell>
                        <Table.Summary.Cell index={1}>
                          <b style={{ color: '#52c41a' }}>¥{totalBuyAmount.toLocaleString()}</b>
                        </Table.Summary.Cell>
                        <Table.Summary.Cell index={2} colSpan={3}></Table.Summary.Cell>
                      </Table.Summary.Row>
                      <Table.Summary.Row>
                        <Table.Summary.Cell index={0} colSpan={3}>
                          <b style={{ color: '#ff4d4f' }}>卖出合计</b>
                        </Table.Summary.Cell>
                        <Table.Summary.Cell index={1}>
                          <b style={{ color: '#ff4d4f' }}>¥{totalSellAmount.toLocaleString()}</b>
                        </Table.Summary.Cell>
                        <Table.Summary.Cell index={2} colSpan={3}></Table.Summary.Cell>
                      </Table.Summary.Row>
                    </>
                  );
                }}
              />
            </Card>
          </Col>
        </Row>
      </div>
    </div>
  );
};

export default TradeRecord;
