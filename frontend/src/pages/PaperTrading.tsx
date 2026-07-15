import { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Spin, Table, Tabs, Tag, TableColumnsType } from 'antd';
import { RiseOutlined, FallOutlined } from '@ant-design/icons';
import { Line } from 'react-chartjs-2';
import axios from 'axios';

const CARD_BG = '#242830';
const CARD_BORDER = '#3a3f4a';
const GOLD = '#e2b04a';
const TEXT_DIM = 'rgba(255,255,255,0.5)';

interface CurvePt { date: string; value: number; }
interface NavResp {
  status: string;
  account: string;
  launchDate: string;
  initCapital: number;
  navCurve: CurvePt[];
  benchmarkCurve: CurvePt[];
  exDivDates: string[];
  benchmarkLabel: string;
  latest: null | {
    date: string; nav: number; benchmark: number; cash: number;
    totalReturn: number | null; benchmarkReturn: number | null;
  };
  caveat: string;
  message?: string;
}
interface Position {
  symbol: string; name: string; shares: number; available: number;
  avgCost: number | null; lastPrice: number | null; priceDate: string | null;
  marketValue: number; pnlPct: number | null;
}
interface PosResp {
  status: string; cash: number; marketValue: number; totalValue: number;
  markDate: string; positions: Position[];
}
interface Trade {
  date: string; symbol: string; name: string; side: string; shares: number;
  price: number; amount: number; cost: number; reason: string;
}

const pctv = (x: number | null) => (x == null ? '—' : `${x >= 0 ? '+' : ''}${x.toFixed(2)}%`);
const col = (x: number | null) => (x == null ? TEXT_DIM : x >= 0 ? '#52c41a' : '#ff4d4f');
const yuan = (x: number) => x.toLocaleString('zh-CN', { maximumFractionDigits: 0 });

const ACCOUNTS = [
  { key: 'A', label: '账户 A · 系统化 Top-K', desc: '每 20 交易日按横截面预测选前 20% 等权持有 (100 万起)' },
  { key: 'B', label: '账户 B · 建议跟随', desc: '镜像真实持仓, 按顾问建议补/减/止损, 自筹现金不注资' },
];

export default function PaperTrading() {
  const [account, setAccount] = useState('A');
  const [loading, setLoading] = useState(true);
  const [nav, setNav] = useState<NavResp | null>(null);
  const [pos, setPos] = useState<PosResp | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [err, setErr] = useState('');

  useEffect(() => { fetchAll(account); }, [account]);

  const fetchAll = async (acct: string) => {
    setLoading(true); setErr('');
    try {
      const [n, p, t] = await Promise.all([
        axios.get(`/api/paper/nav?account=${acct}`),
        axios.get(`/api/paper/positions?account=${acct}`),
        axios.get(`/api/paper/trades?account=${acct}`),
      ]);
      if (n.data.status === 'success') setNav(n.data);
      else { setErr(n.data.message || '纸面账户未就绪'); setNav(null); }
      setPos(p.data.status === 'success' ? p.data : null);
      setTrades(t.data.status === 'success' ? t.data.trades : []);
    } catch { setErr('无法连接到服务器'); }
    setLoading(false);
  };

  const navChart = nav && {
    labels: nav.navCurve.map((p) => p.date),
    datasets: [
      { label: '纸面 NAV', data: nav.navCurve.map((p) => p.value), borderColor: '#52c41a', backgroundColor: 'rgba(82,196,26,0.08)', borderWidth: 2, pointRadius: nav.navCurve.length < 40 ? 2 : 0, tension: 0.1, fill: true },
      { label: nav.benchmarkLabel, data: nav.benchmarkCurve.map((p) => p.value), borderColor: 'rgba(255,255,255,0.45)', borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, tension: 0.1 },
    ],
  };
  const lineOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: 'rgba(255,255,255,0.7)', boxWidth: 14 } } },
    scales: {
      x: { ticks: { color: 'rgba(255,255,255,0.4)', maxTicksLimit: 12 }, grid: { color: '#2f333c' } },
      y: { ticks: { color: 'rgba(255,255,255,0.4)' }, grid: { color: '#2f333c' } },
    },
  };

  const posCols: TableColumnsType<Position> = [
    { title: '代码', dataIndex: 'symbol', width: 110 },
    { title: '名称', dataIndex: 'name', width: 120 },
    { title: '股数', dataIndex: 'shares', align: 'right', render: (x: number) => yuan(x) },
    { title: '可卖', dataIndex: 'available', align: 'right', render: (x: number, r) => <span style={{ color: x < r.shares ? '#faad14' : undefined }}>{yuan(x)}</span> },
    { title: '成本', dataIndex: 'avgCost', align: 'right', render: (x: number | null) => (x == null ? '—' : x.toFixed(3)) },
    { title: '现价', dataIndex: 'lastPrice', align: 'right', render: (x: number | null) => (x == null ? '—' : x.toFixed(3)) },
    { title: '市值', dataIndex: 'marketValue', align: 'right', render: (x: number) => `¥${yuan(x)}` },
    { title: '浮盈', dataIndex: 'pnlPct', align: 'right', render: (x: number | null) => <span style={{ color: col(x) }}>{pctv(x)}</span> },
  ];
  const tradeCols: TableColumnsType<Trade> = [
    { title: '成交日', dataIndex: 'date', width: 110 },
    { title: '代码', dataIndex: 'symbol', width: 110 },
    { title: '名称', dataIndex: 'name', width: 110 },
    { title: '方向', dataIndex: 'side', width: 70, render: (s: string) => <Tag color={s === 'buy' ? 'red' : 'green'}>{s === 'buy' ? '买入' : '卖出'}</Tag> },
    { title: '股数', dataIndex: 'shares', align: 'right', render: (x: number) => yuan(x) },
    { title: '价格', dataIndex: 'price', align: 'right', render: (x: number) => x.toFixed(3) },
    { title: '金额', dataIndex: 'amount', align: 'right', render: (x: number) => `¥${yuan(x)}` },
    { title: '成本', dataIndex: 'cost', align: 'right', render: (x: number) => `¥${x.toFixed(0)}` },
    { title: '原因', dataIndex: 'reason', width: 130 },
  ];

  const meta = ACCOUNTS.find((a) => a.key === account)!;

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#1e2229', padding: 24 }}>
      <div style={{ maxWidth: 1400, margin: '0 auto' }}>
        <h1 style={{ color: '#e0e0e0', fontSize: 22, marginBottom: 4 }}>纸面交易 (前瞻验证)</h1>
        <p style={{ color: TEXT_DIM, fontSize: 13, marginBottom: 16 }}>
          信号 as-of 收盘冻结 · D+1 开盘成交 · 已扣真实成本 · T+1 约束 —— 用真实前瞻记账检验"信号能不能赚钱",
          区别于有幸存者偏差的历史回测。
        </p>

        <Tabs
          activeKey={account}
          onChange={setAccount}
          items={ACCOUNTS.map((a) => ({ key: a.key, label: a.label }))}
        />
        <p style={{ color: TEXT_DIM, fontSize: 12, marginTop: -8, marginBottom: 16 }}>{meta.desc}</p>

        {loading ? <Spin tip="加载纸面账户..." style={{ display: 'block', margin: '60px auto' }} /> : !nav ? (
          <Card style={{ textAlign: 'center', padding: 40, background: CARD_BG, border: `1px solid ${CARD_BORDER}` }}>
            <p style={{ color: TEXT_DIM }}>{err || '暂无纸面数据 (主机尚未 --init / --advance)'}</p>
          </Card>
        ) : (
          <>
            {/* caveat */}
            <div style={{ background: 'rgba(226,176,74,0.08)', border: `1px solid ${GOLD}`, borderRadius: 6, padding: '10px 14px', marginBottom: 16, color: GOLD, fontSize: 13, lineHeight: 1.6 }}>
              ⚠️ {nav.caveat}
              <div style={{ color: TEXT_DIM, marginTop: 4, fontSize: 12 }}>
                纸面启动日 {nav.launchDate}
                {nav.latest ? ` · 最新 ${nav.latest.date}` : ''}
                {nav.exDivDates.length > 0 && (
                  <span style={{ color: '#ff7a45' }}> · ⚑ 持有期除权除息(未复权失真): {nav.exDivDates.join(', ')}</span>
                )}
              </div>
            </div>

            {/* headline */}
            <Card style={{ marginBottom: 16, background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 14 } }}>
              <Row gutter={16}>
                <Col span={5}><Statistic title={<span style={{ color: TEXT_DIM }}>总资产</span>} value={pos ? yuan(pos.totalValue) : '—'} prefix="¥" valueStyle={{ color: '#e0e0e0', fontSize: 22 }} /></Col>
                <Col span={5}><Statistic title={<span style={{ color: TEXT_DIM }}>纸面收益</span>} value={nav.latest?.totalReturn != null ? nav.latest.totalReturn * 100 : 0} precision={2} suffix="%" valueStyle={{ color: col(nav.latest?.totalReturn ?? 0), fontSize: 22 }} prefix={(nav.latest?.totalReturn ?? 0) >= 0 ? <RiseOutlined /> : <FallOutlined />} /></Col>
                <Col span={5}><Statistic title={<span style={{ color: TEXT_DIM }}>基准收益</span>} value={nav.latest?.benchmarkReturn != null ? nav.latest.benchmarkReturn * 100 : 0} precision={2} suffix="%" valueStyle={{ color: 'rgba(255,255,255,0.75)', fontSize: 22 }} /></Col>
                <Col span={5}><Statistic title={<span style={{ color: TEXT_DIM }}>超额 (纸面 − 基准)</span>} value={nav.latest ? ((nav.latest.totalReturn ?? 0) - (nav.latest.benchmarkReturn ?? 0)) * 100 : 0} precision={2} suffix="%" valueStyle={{ color: col(nav.latest ? (nav.latest.totalReturn ?? 0) - (nav.latest.benchmarkReturn ?? 0) : 0), fontSize: 22 }} /></Col>
                <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM }}>现金</span>} value={pos ? yuan(pos.cash) : '—'} prefix="¥" valueStyle={{ color: 'rgba(255,255,255,0.75)', fontSize: 18 }} /></Col>
              </Row>
            </Card>

            {/* NAV 曲线 */}
            <Card title={<span style={{ color: '#52c41a' }}>纸面净值 vs 基准 (起点 = 1.0)</span>} style={{ marginBottom: 16, background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 12 } }}>
              {navChart && nav.navCurve.length > 0 ? (
                <div style={{ height: 320 }}><Line data={navChart} options={lineOpts} /></div>
              ) : (
                <p style={{ color: TEXT_DIM, textAlign: 'center', padding: 40 }}>
                  纸面刚启动 (启动日 {nav.launchDate}), 曲线将随每日 --advance 累积。
                </p>
              )}
            </Card>

            <Row gutter={16}>
              <Col span={11}>
                <Card title="当前持仓" size="small" style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 8 } }}>
                  <Table columns={posCols} dataSource={pos?.positions || []} rowKey="symbol" pagination={false} size="small" />
                </Card>
              </Col>
              <Col span={13}>
                <Card title="成交流水 (最近在前)" size="small" style={{ background: CARD_BG, border: `1px solid ${CARD_BORDER}` }} styles={{ body: { padding: 8 } }}>
                  <Table columns={tradeCols} dataSource={trades} rowKey={(r) => `${r.date}-${r.symbol}-${r.side}-${r.shares}`} pagination={{ pageSize: 10, size: 'small' }} size="small" locale={{ emptyText: '暂无成交 (账户A待首个调仓日; 账户B待建议触发)' }} />
                </Card>
              </Col>
            </Row>
          </>
        )}
      </div>
    </div>
  );
}
