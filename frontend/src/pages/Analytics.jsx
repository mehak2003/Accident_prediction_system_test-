import { useEffect, useState } from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, RadarChart,
  PolarGrid, PolarAngleAxis, Radar,
} from 'recharts';
import ChartCard from '../components/ChartCard';
import { fetchEda, fetchModelMetrics } from '../api';
import { mockEda, mockMetrics } from '../mockData';

const COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#3b82f6', '#ec4899', '#8b5cf6', '#14b8a6'];

export default function Analytics() {
  const [data, setData] = useState(mockEda);
  const [metrics, setMetrics] = useState(mockMetrics);
  const [live, setLive] = useState(false);

  useEffect(() => {
    const endpoints = ['hourly', 'weekly', 'severity', 'weather', 'top_areas', 'collision_types', 'causes', 'severity_by_weather'];
    Promise.allSettled([
      ...endpoints.map(e => fetchEda(e).then(r => ({ key: e, data: r.data }))),
      fetchModelMetrics().then(r => r.data),
    ]).then(results => {
      const d = {};
      let anyLive = false;
      results.slice(0, endpoints.length).forEach(r => {
        if (r.status === 'fulfilled') { d[r.value.key] = r.value.data; anyLive = true; }
      });
      if (anyLive) setData(prev => ({ ...prev, ...d }));
      const metricsResult = results[endpoints.length];
      if (metricsResult.status === 'fulfilled') setMetrics(metricsResult.value);
      if (anyLive) setLive(true);
    });
  }, []);

  const toArr = (obj) => obj ? Object.entries(obj).map(([name, value]) => ({ name, value })) : [];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold" style={{ color: 'var(--clr-text)' }}>Analytics</h2>
        <p className="text-sm" style={{ color: 'var(--clr-text-muted)' }}>
          Exploratory Data Analysis & Model Performance
          {!live && <span className="ml-2 text-xs px-2 py-0.5 rounded-full" style={{ background: '#f59e0b20', color: '#f59e0b' }}>Sample Data</span>}
        </p>
      </div>

      {/* Model metrics */}
      {metrics && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="rounded-xl p-5 border text-center"
            style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
            <p className="text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--clr-text-muted)' }}>Accuracy</p>
            <p className="text-3xl font-bold" style={{ color: 'var(--clr-success)' }}>
              {(metrics.accuracy * 100).toFixed(1)}%
            </p>
          </div>
          <div className="rounded-xl p-5 border text-center"
            style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
            <p className="text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--clr-text-muted)' }}>Model</p>
            <p className="text-lg font-bold" style={{ color: 'var(--clr-primary-light)' }}>Random Forest</p>
            <p className="text-xs" style={{ color: 'var(--clr-text-muted)' }}>200 estimators</p>
          </div>
          <div className="rounded-xl p-5 border text-center"
            style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
            <p className="text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--clr-text-muted)' }}>Train/Test Split</p>
            <p className="text-3xl font-bold" style={{ color: 'var(--clr-info)' }}>80/20</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Hourly distribution */}
        {data.hourly && (
          <ChartCard title="Accidents by Hour of Day">
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={toArr(data.hourly)}>
                <XAxis dataKey="name" tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)', borderRadius: 8, color: 'var(--clr-text)' }} />
                <Line type="monotone" dataKey="value" stroke="#6366f1" strokeWidth={2} dot={{ r: 3, fill: '#6366f1' }} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        )}

        {/* Weekly distribution */}
        {data.weekly && (
          <ChartCard title="Accidents by Day of Week">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={toArr(data.weekly)}>
                <XAxis dataKey="name" tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)', borderRadius: 8, color: 'var(--clr-text)' }} />
                <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}

        {/* Severity */}
        {data.severity && (
          <ChartCard title="Severity Distribution">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={toArr(data.severity)} cx="50%" cy="50%" innerRadius={50} outerRadius={80}
                  paddingAngle={3} dataKey="value" stroke="none">
                  {toArr(data.severity).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)', borderRadius: 8, color: 'var(--clr-text)' }} />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap gap-3 justify-center">
              {toArr(data.severity).map((d, i) => (
                <span key={d.name} className="flex items-center gap-1.5 text-xs" style={{ color: 'var(--clr-text-muted)' }}>
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />
                  {d.name}
                </span>
              ))}
            </div>
          </ChartCard>
        )}

        {/* Weather */}
        {data.weather && (
          <ChartCard title="Accidents by Weather">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={toArr(data.weather)} layout="vertical">
                <XAxis type="number" tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} width={100} />
                <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)', borderRadius: 8, color: 'var(--clr-text)' }} />
                <Bar dataKey="value" fill="#f59e0b" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}

        {/* Top areas */}
        {data.top_areas && (
          <ChartCard title="Top Accident Areas">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={toArr(data.top_areas).slice(0, 8)} layout="vertical">
                <XAxis type="number" tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" tick={{ fill: 'var(--clr-text-muted)', fontSize: 10 }} axisLine={false} tickLine={false} width={120} />
                <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)', borderRadius: 8, color: 'var(--clr-text)' }} />
                <Bar dataKey="value" fill="#ec4899" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}

        {/* Collision types */}
        {data.collision_types && (
          <ChartCard title="Collision Types">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={toArr(data.collision_types).slice(0, 6)} layout="vertical">
                <XAxis type="number" tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" tick={{ fill: 'var(--clr-text-muted)', fontSize: 10 }} axisLine={false} tickLine={false} width={140} />
                <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)', borderRadius: 8, color: 'var(--clr-text)' }} />
                <Bar dataKey="value" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}

        {/* Causes */}
        {data.causes && (
          <ChartCard title="Top Accident Causes" className="lg:col-span-2">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={toArr(data.causes).slice(0, 10)}>
                <XAxis dataKey="name" tick={{ fill: 'var(--clr-text-muted)', fontSize: 10, angle: -30 }} axisLine={false} tickLine={false} height={60} />
                <YAxis tick={{ fill: 'var(--clr-text-muted)', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)', borderRadius: 8, color: 'var(--clr-text)' }} />
                <Bar dataKey="value" fill="#14b8a6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}
      </div>
    </div>
  );
}
