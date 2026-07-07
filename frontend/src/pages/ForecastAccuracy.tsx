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
import { Button, Select, Card, Statistic, Row, Col, Tag, Spin, message, Space } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined, RiseOutlined, FallOutlined, AimOutlined } from '@ant-design/icons';
import { Link } from 'react-router-dom';
import axios from 'axios';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

import { stockList } from '../constants/stocks';

interface SeriesPt { date: string; pred: number; actual: number; }
interface Oos {
  n: number;
  dir_acc: number;
  hit_rate_up: number | null;
  mean_ret_up_net: number | null;
  series: SeriesPt[];
}
interface Current {
  dataDate: string;
  lastPrice: number;
  rsi: number;
  candidate: boolean;
  ret20Pred: number;
  upProb: number;
  tpProb: number;
  tpPrice: number;
  slPrice: number;
  verdict: string;
}
interface PredictData {
  status: string;
  symbol: string;
  horizon: number;
  trainDate: string;
  a2Usable: boolean;
  a3Usable: boolean;
  current: Current | null;
  oos: Oos | null;
  caveat: string;
}

const chartPosition = 'top' as const;
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

const pct = (x: number | null | undefined, d = 1) =>
  x === null || x === undefined ? '—' : `${x >= 0 ? '+' : ''}${(x * 100).toFixed(d)}%`;

export default function ForecastAccuracy() {
  const [symbol, setSymbol] = useState('300124.SZ');
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<PredictData | null>(null);
  const [showAll, setShowAll] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`/api/advisor/predict/${symbol}`);
      if (res.data.status === 'success') {
        setData(res.data as PredictData);
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
  }, [symbol]);

  const oos = data?.oos ?? null;
  const cur = data?.current ?? null;
  const dirAcc = oos ? oos.dir_acc * 100 : 0;

  // 方向命中 = 预测涨跌方向与实际一致 (基于可见的月度下采样序列, 与下方表格口径一致)
  const hitOf = (arr: SeriesPt[]): number | null => {
    if (!arr.length) return null;
    const h = arr.filter((p) => (p.pred >= 0) === (p.actual >= 0)).length;
    return h / arr.length;
  };
  const recentHit = (n: number) => (oos ? hitOf(oos.series.slice(-n)) : null);
  const hit6 = recentHit(6);
  const hit12 = recentHit(12);
  const hit24 = recentHit(24);
  // 最新在前, 供表格 & 卡片
  const seriesRev = oos ? [...oos.series].reverse() : [];
  const tableRows = showAll ? seriesRev : seriesRev.slice(0, 24);
  const hitColor = (h: number | null) =>
    h === null ? TEXT_DIM : h >= 0.55 ? '#3fb950' : h >= 0.5 ? GOLD : '#f85149';

  // 预测 20 日收益 vs 实际 20 日收益 (样本外, 月度下采样)
  const seriesChart = oos && oos.series.length ? {
    labels: oos.series.map((p) => p.date.slice(0, 7)),
    datasets: [
      {
        label: '预测 20 日收益',
        data: oos.series.map((p) => p.pred * 100),
        borderColor: '#f0883e',
        backgroundColor: 'rgba(240,136,62,0.08)',
        fill: false,
        pointRadius: 1.5,
        borderWidth: 2,
        borderDash: [4, 2],
      },
      {
        label: '实际 20 日收益',
        data: oos.series.map((p) => p.actual * 100),
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.08)',
        fill: true,
        pointRadius: 1.5,
        borderWidth: 2,
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

  const seriesChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: { display: true, text: `样本外 预测 vs 实际 20 日收益 — ${symbol}`, color: TEXT_LIGHT, font: { size: 14 } },
      legend: { position: chartPosition, labels: { color: TEXT_DIM } },
    },
    scales: {
      x: { ticks: { color: TEXT_DIM, maxTicksLimit: 14 }, grid: { color: CARD_BORDER } },
      y: { title: { display: true, text: '收益 (%)', color: TEXT_DIM }, ticks: { color: TEXT_DIM, callback: (v: any) => v + '%' }, grid: { color: CARD_BORDER } },
    },
  };

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(180deg, #1e2229 0%, #1a1e25 100%)' }}>
      <div style={{ background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)', padding: '14px 32px', color: 'white', display: 'flex', justifyContent: 'space-between', alignItems: 'center', boxShadow: '0 4px 20px rgba(0,0,0,0.3)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <div>
          <h2 style={{ margin: 0, fontSize: 20, fontWeight: 600, letterSpacing: 1 }}>
            <AimOutlined style={{ marginRight: 10, color: GOLD }} />
            预测准确性验证 (样本外 walk-forward)
          </h2>
        </div>
        <Space>
          <Link to="/" style={{ color: 'rgba(255,255,255,0.8)', textDecoration: 'none', padding: '8px 16px', background: 'rgba(226,176,74,0.15)', borderRadius: 6, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid rgba(226,176,74,0.3)' }}>← 返回主页</Link>
          <Link to="/trade" style={{ color: 'rgba(255,255,255,0.8)', textDecoration: 'none', padding: '8px 16px', background: 'rgba(226,176,74,0.15)', borderRadius: 6, fontSize: 13, display: 'flex', alignItems: 'center', gap: 6, border: '1px solid rgba(226,176,74,0.3)' }}>交易记录</Link>
        </Space>
      </div>

      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 24px 48px' }}>
        {/* 诚实 caveat */}
        <div style={{ background: 'rgba(226,176,74,0.08)', border: `1px solid ${GOLD}`, borderRadius: 6, padding: '10px 14px', marginBottom: 16, color: GOLD, fontSize: 13, lineHeight: 1.6 }}>
          ⚠️ {data?.caveat || 'edge 薄(横截面 rank-IC≈0.05); 单只择时不如买入持有, 仅作方向参考。已扣成本。'}
          <div style={{ color: TEXT_DIM, marginTop: 4, fontSize: 12 }}>
            这里展示的是【样本外(OOS)】真实预测 vs 实际, 非拟合训练集。方向准确率略高于 50% 即为薄 edge, 别期待高胜率。
          </div>
        </div>

        {/* 控制栏 */}
        <Card style={{ ...darkCardStyle, marginBottom: 20 }} styles={{ body: { padding: '12px 16px' } }}>
          <Row gutter={24} align="middle">
            <Col span={8}>
              <div style={{ color: TEXT_DIM, fontSize: 12, marginBottom: 4 }}>选择股票</div>
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
            <Col span={4}>
              <Button type="primary" onClick={fetchData} loading={loading} icon={<RiseOutlined />} style={{ background: GOLD, borderColor: GOLD, marginTop: 16 }}>
                验证
              </Button>
            </Col>
            <Col span={12}>
              {data && (
                <div style={{ color: TEXT_DIM, fontSize: 12, marginTop: 16 }}>
                  {oos ? `${oos.n} 条 OOS 样本 · 预测周期 ${data.horizon} 交易日` : '该标的不在 A 股训练池 (港股/ETF 无 OOS 历史)'}
                </div>
              )}
            </Col>
          </Row>
        </Card>

        {loading && <Spin tip="加载样本外验证..." style={{ display: 'block', margin: '60px auto' }} />}

        {data && !loading && (
          <>
            {/* 最近方向命中率 — 一眼看懂近端成功率 */}
            {oos ? (
              <>
                <Row gutter={12} style={{ marginBottom: 12 }}>
                  <Col span={6}>
                    <Card style={{ ...darkCardStyle, borderColor: GOLD }} styles={{ body: { padding: '14px 16px' } }}>
                      <Statistic
                        title={<span style={{ color: GOLD, fontSize: 12, fontWeight: 600 }}>近 6 次方向命中</span>}
                        value={hit6 === null ? 0 : hit6 * 100}
                        precision={0}
                        suffix={<span style={{ fontSize: 13, color: TEXT_DIM }}>{hit6 === null ? '' : ` (${Math.round(hit6 * Math.min(6, oos.series.length))}/${Math.min(6, oos.series.length)})`}</span>}
                        valueStyle={{ color: hitColor(hit6), fontSize: 28, fontWeight: 700 }}
                        prefix={(hit6 ?? 0) >= 0.5 ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
                      />
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card style={darkCardStyle} styles={{ body: { padding: '14px 16px' } }}>
                      <Statistic
                        title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>近 12 次方向命中</span>}
                        value={hit12 === null ? 0 : hit12 * 100}
                        precision={0}
                        suffix={<span style={{ fontSize: 13, color: TEXT_DIM }}>{hit12 === null ? '' : ` (${Math.round(hit12 * Math.min(12, oos.series.length))}/${Math.min(12, oos.series.length)})`}</span>}
                        valueStyle={{ color: hitColor(hit12), fontSize: 28, fontWeight: 700 }}
                      />
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card style={darkCardStyle} styles={{ body: { padding: '14px 16px' } }}>
                      <Statistic
                        title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>近 24 次方向命中</span>}
                        value={hit24 === null ? 0 : hit24 * 100}
                        precision={0}
                        suffix={<span style={{ fontSize: 13, color: TEXT_DIM }}>{hit24 === null ? '' : ` (${Math.round(hit24 * Math.min(24, oos.series.length))}/${Math.min(24, oos.series.length)})`}</span>}
                        valueStyle={{ color: hitColor(hit24), fontSize: 28, fontWeight: 700 }}
                      />
                    </Card>
                  </Col>
                  <Col span={6}>
                    <Card style={darkCardStyle} styles={{ body: { padding: '14px 16px' } }}>
                      <Statistic
                        title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>全样本方向准确率(日频)</span>}
                        value={dirAcc}
                        precision={1}
                        suffix="%"
                        valueStyle={{ color: dirAcc >= 52 ? '#3fb950' : dirAcc >= 50 ? GOLD : '#f85149', fontSize: 28, fontWeight: 700 }}
                      />
                    </Card>
                  </Col>
                </Row>
                <Row gutter={12} style={{ marginBottom: 20 }}>
                  <Col span={8}>
                    <Card style={darkCardStyle} styles={{ body: { padding: '10px 16px' } }}>
                      <Statistic
                        title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>看涨命中率 (预测涨→实际涨)</span>}
                        value={oos.hit_rate_up === null ? 0 : oos.hit_rate_up * 100}
                        precision={1}
                        suffix="%"
                        valueStyle={{ color: '#58a6ff', fontSize: 20 }}
                      />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card style={darkCardStyle} styles={{ body: { padding: '10px 16px' } }}>
                      <Statistic
                        title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>预测涨·平均净收益/笔</span>}
                        value={oos.mean_ret_up_net === null ? 0 : oos.mean_ret_up_net * 100}
                        precision={2}
                        suffix="%"
                        valueStyle={{ color: (oos.mean_ret_up_net ?? 0) >= 0 ? '#3fb950' : '#f85149', fontSize: 20 }}
                      />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card style={darkCardStyle} styles={{ body: { padding: '10px 16px' } }}>
                      <Statistic
                        title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>OOS 样本数 (日频)</span>}
                        value={oos.n}
                        valueStyle={{ color: TEXT_LIGHT, fontSize: 20 }}
                      />
                    </Card>
                  </Col>
                </Row>
                <div style={{ color: TEXT_DIM, fontSize: 12, marginBottom: 20, lineHeight: 1.6 }}>
                  💡 「近 N 次」按最新的月度采样点算方向命中(涨/跌方向对不对)。<b style={{ color: GOLD }}>单只近端命中率波动极大、常低于 50%</b> —
                  这正是模型的诚实面: edge 在<b>横截面排序</b>(见「策略回测」), 不是单只择时。别用单只近端命中率下重注。
                </div>
              </>
            ) : (
              <Card style={{ ...darkCardStyle, marginBottom: 20 }} styles={{ body: { padding: 24, textAlign: 'center' } }}>
                <span style={{ color: TEXT_DIM }}>该标的不在 A 股训练池, 无样本外历史。下方仅显示当前信号 (若模型可打分)。</span>
              </Card>
            )}

            {/* 当前信号 */}
            {cur && (
              <Card
                title={<span style={{ color: GOLD }}>当前 {data.horizon} 日信号 · 数据截至 {cur.dataDate}</span>}
                style={{ ...darkCardStyle, marginBottom: 20 }}
                styles={{ body: { padding: '14px 16px' } }}
              >
                <Row gutter={16}>
                  <Col span={5}><Statistic title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>预测 20 日收益</span>} value={cur.ret20Pred * 100} precision={2} suffix="%" valueStyle={{ color: cur.ret20Pred >= 0 ? '#3fb950' : '#f85149', fontSize: 20 }} prefix={cur.ret20Pred >= 0 ? <RiseOutlined /> : <FallOutlined />} /></Col>
                  <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>上涨概率</span>} value={cur.upProb * 100} precision={0} suffix="%" valueStyle={{ color: '#58a6ff', fontSize: 20 }} /></Col>
                  <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>P(先触止盈)</span>} value={cur.tpProb * 100} precision={0} suffix="%" valueStyle={{ color: TEXT_LIGHT, fontSize: 20 }} /></Col>
                  <Col span={4}><Statistic title={<span style={{ color: TEXT_DIM, fontSize: 12 }}>止盈 / 止损</span>} value={cur.tpPrice} precision={2} suffix={` / ${cur.slPrice}`} valueStyle={{ color: TEXT_LIGHT, fontSize: 16 }} /></Col>
                  <Col span={7}>
                    <div style={{ color: TEXT_DIM, fontSize: 12, marginBottom: 4 }}>建议</div>
                    <div>
                      {cur.candidate ? <Tag color="orange">补仓候选态</Tag> : <Tag>非候选态</Tag>}
                      <span style={{ color: TEXT_LIGHT, fontSize: 13 }}>RSI {cur.rsi}</span>
                    </div>
                    <div style={{ color: TEXT_LIGHT, fontSize: 12, marginTop: 4 }}>{cur.verdict}</div>
                  </Col>
                </Row>
              </Card>
            )}

            {/* 预测 vs 实际曲线 */}
            {seriesChart && (
              <Card style={{ ...darkCardStyle, marginBottom: 20 }} styles={{ body: { padding: '12px 16px' } }}>
                <div style={{ height: 320 }}>
                  <Line data={seriesChart} options={seriesChartOptions} plugins={[darkChartPlugin]} />
                </div>
                <div style={{ color: TEXT_DIM, fontSize: 11, marginTop: 8, textAlign: 'center' }}>
                  橙虚线=模型预测的 20 日收益 · 蓝=实际实现的 20 日收益 · 两线同向多即方向 edge 成立 (月度下采样)
                </div>
              </Card>
            )}

            {/* 逐条明细 — 最新在前 */}
            {oos && oos.series.length > 0 && (
              <Card style={darkCardStyle} styles={{ body: { padding: '8px 12px' } }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <span style={{ color: TEXT_DIM, fontSize: 12 }}>
                    逐条明细(最新在前) · 共 {seriesRev.length} 条{showAll ? '' : `,默认显示最近 ${Math.min(24, seriesRev.length)} 条`}
                  </span>
                  {seriesRev.length > 24 && (
                    <Button size="small" type="text" style={{ color: GOLD }} onClick={() => setShowAll((s) => !s)}>
                      {showAll ? '收起' : '展开全部'}
                    </Button>
                  )}
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, color: TEXT_LIGHT }}>
                  <thead>
                    <tr style={{ borderBottom: `2px solid ${GOLD}` }}>
                      <th style={{ padding: 8, textAlign: 'left' }}>日期</th>
                      <th style={{ padding: 8 }}>预测 20 日</th>
                      <th style={{ padding: 8 }}>实际 20 日</th>
                      <th style={{ padding: 8 }}>方向</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableRows.map((p) => {
                      const same = (p.pred >= 0) === (p.actual >= 0);
                      return (
                        <tr key={p.date} style={{ borderBottom: `1px solid ${CARD_BORDER}` }}>
                          <td style={{ padding: 6 }}>{p.date}</td>
                          <td style={{ padding: 6, textAlign: 'center', color: p.pred >= 0 ? '#3fb950' : '#f85149' }}>{pct(p.pred, 2)}</td>
                          <td style={{ padding: 6, textAlign: 'center', color: p.actual >= 0 ? '#3fb950' : '#f85149' }}>{pct(p.actual, 2)}</td>
                          <td style={{ padding: 6, textAlign: 'center' }}>
                            {same ? <Tag color="green">✓</Tag> : <Tag color="red">✗</Tag>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </Card>
            )}
          </>
        )}
      </div>
    </div>
  );
}
