import React, { useState, useEffect } from 'react';
import { Table, Card, Statistic, Row, Col, Tag, Button, Modal, Form, InputNumber, Input, message } from 'antd';
import { RiseOutlined, FallOutlined, PlusOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons';
import axios from 'axios';

const PositionManager: React.FC = () => {
  const [positions, setPositions] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [modalVisible, setModalVisible] = useState<boolean>(false);
  const [editingPosition, setEditingPosition] = useState<any>(null);
  const [advisor, setAdvisor] = useState<Record<string, any>>({});
  const [advisorMeta, setAdvisorMeta] = useState<any>(null);
  const [form] = Form.useForm();

  useEffect(() => {
    fetchPositions();
    fetchAdvisor();
  }, []);

  const fetchPositions = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/positions');
      if (response.data.status === 'success') {
        setPositions(response.data.positions);
      }
    } catch (error) {
      message.error('获取持仓失败');
    }
    setLoading(false);
  };

  const fetchAdvisor = async () => {
    try {
      const res = await axios.get('/api/advisor/holdings');
      if (res.data.status === 'success') {
        const map: Record<string, any> = {};
        (res.data.holdings || []).forEach((h: any) => { map[h.symbol] = h; });
        setAdvisor(map);
        setAdvisorMeta(res.data);
      } else {
        setAdvisorMeta({ error: res.data.message });
      }
    } catch (error) {
      setAdvisorMeta({ error: '补仓顾问接口不可用' });
    }
  };

  const handleAdd = () => {
    setEditingPosition(null);
    form.resetFields();
    setModalVisible(true);
  };

  const handleEdit = (record: any) => {
    setEditingPosition(record);
    form.setFieldsValue(record);
    setModalVisible(true);
  };

  const handleDelete = async (symbol: string) => {
    try {
      await axios.delete(`/api/positions/${symbol}`);
      message.success('删除成功');
      fetchPositions();
    } catch (error) {
      message.error('删除失败');
    }
  };

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      if (editingPosition) {
        await axios.put(`/api/positions/${editingPosition.symbol}`, values);
        message.success('更新成功');
      } else {
        await axios.post('/api/positions', values);
        message.success('添加成功');
      }
      setModalVisible(false);
      fetchPositions();
    } catch (error) {
      message.error('操作失败');
    }
  };

  const columns = [
    { title: '股票代码', dataIndex: 'symbol', key: 'symbol' },
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '股数', dataIndex: 'shares', key: 'shares', render: (v: number) => v.toLocaleString() },
    { title: '成本价', dataIndex: 'cost', key: 'cost', render: (v: number) => `¥${v?.toFixed(2) || 0}` },
    { title: '现价', dataIndex: 'current', key: 'current', render: (v: number) => `¥${v?.toFixed(2) || 0}` },
    { 
      title: '盈亏', 
      dataIndex: 'profit', 
      key: 'profit',
      render: (v: number) => (
        <Statistic 
          value={v} 
          precision={2} 
          valueStyle={{ color: v >= 0 ? '#3f8600' : '#cf1322' }}
          prefix={v >= 0 ? <RiseOutlined /> : <FallOutlined />}
        />
      )
    },
    { 
      title: '盈亏%', 
      dataIndex: 'profit_pct', 
      key: 'profit_pct',
      render: (v: number) => (
        <Tag color={v >= 0 ? 'green' : 'red'}>{v?.toFixed(1) || 0}%</Tag>
      )
    },
    {
      title: '补仓顾问(ML)',
      key: 'advisor',
      width: 320,
      render: (record: any) => {
        const a = advisor[record.symbol];
        if (!a || !a.ready) return <Tag>模型未就绪</Tag>;
        return (
          <div style={{ fontSize: 12, lineHeight: 1.5 }}>
            <div>
              {a.candidate
                ? <Tag color="orange">补仓候选态</Tag>
                : <Tag>非候选态</Tag>}
              <span>RSI {a.rsi}</span>
            </div>
            <div>方案2: 20日 {(a.ret20Pred * 100).toFixed(1)}% · 涨概率 {(a.upProb * 100).toFixed(0)}%</div>
            <div>方案3: P(止盈) {(a.tpProb * 100).toFixed(0)}% · 止盈 {a.tpPrice} / 止损 {a.slPrice}</div>
            <div style={{ color: '#555', marginTop: 2 }}>{a.verdict}</div>
          </div>
        );
      }
    },
    {
      title: '操作',
      key: 'action',
      render: (record: any) => (
        <>
          <Button type="link" icon={<EditOutlined />} onClick={() => handleEdit(record)}>编辑</Button>
          <Button type="link" danger icon={<DeleteOutlined />} onClick={() => handleDelete(record.symbol)}>删除</Button>
        </>
      )
    }
  ];

  // 计算汇总
  const totalProfit = positions.reduce((sum, p) => sum + p.profit * p.shares, 0);
  const totalCost = positions.reduce((sum, p) => sum + p.cost * p.shares, 0);
  const totalValue = positions.reduce((sum, p) => sum + p.current * p.shares, 0);

  return (
    <Card title="持仓管理">
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} md={6}>
          <Statistic title="持仓市值" value={totalValue} precision={0} prefix="¥" />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Statistic title="总成本" value={totalCost} precision={0} prefix="¥" />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Statistic 
            title="总盈亏" 
            value={totalProfit} 
            precision={0} 
            prefix="¥"
            valueStyle={{ color: totalProfit >= 0 ? '#3f8600' : '#cf1322' }}
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Statistic 
            title="收益率" 
            value={totalProfit / totalCost * 100} 
            precision={1}
            suffix="%"
            valueStyle={{ color: totalProfit >= 0 ? '#3f8600' : '#cf1322' }}
          />
        </Col>
      </Row>

      {advisorMeta && (
        <div style={{ marginBottom: 12, padding: '8px 12px', background: '#fffbe6',
                      border: '1px solid #ffe58f', borderRadius: 4, fontSize: 12 }}>
          {advisorMeta.error
            ? <span>⚠️ 补仓顾问: {advisorMeta.error}</span>
            : <span>
                🩺 补仓顾问(ML) · 模型训练截至 {advisorMeta.cutoff} · 预测周期 {advisorMeta.horizon} 交易日 ·
                方案2 {advisorMeta.a2Usable ? '✅' : '❌薄'} 方案3 {advisorMeta.a3Usable ? '✅' : '❌薄'}
                <br/>{advisorMeta.caveat} — 模型仅辅助排序, 补仓与否以纪律(破位止损/不接飞刀)为先。
              </span>}
        </div>
      )}

      <Button type="primary" icon={<PlusOutlined />} onClick={handleAdd} style={{ marginBottom: 16 }}>
        添加持仓
      </Button>

      <div className="table-scroll-wrapper"><Table 
        dataSource={positions} 
        columns={columns} 
        rowKey="symbol"
        loading={loading}
        pagination={false}
      />

      </div>
      <Modal
        title={editingPosition ? '编辑持仓' : '添加持仓'}
        visible={modalVisible}
        onOk={handleSubmit}
        onCancel={() => setModalVisible(false)}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="symbol" label="股票代码" rules={[{ required: true }]}>
            <Input placeholder="如 300124.SZ" disabled={editingPosition} />
          </Form.Item>
          <Form.Item name="name" label="股票名称" rules={[{ required: true }]}>
            <Input placeholder="如 汇川技术" />
          </Form.Item>
          <Form.Item name="shares" label="股数" rules={[{ required: true }]}>
            <InputNumber min={1} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="cost" label="成本价" rules={[{ required: true }]}>
            <InputNumber min={0.01} precision={3} style={{ width: '100%' }} />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  );
};

export default PositionManager;