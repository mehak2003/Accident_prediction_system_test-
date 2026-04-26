import { useEffect, useState, memo, useCallback } from 'react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell,
} from 'recharts';
import ChartCard from '../components/ChartCard';
import { fetchEda, fetchModelMetrics } from '../api';

const COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#3b82f6', '#ec4899', '#8b5cf6', '#14b8a6'];

// Polyfill for requestIdleCallback (performance optimization - non-visual)
const requestIdleCallback = window.requestIdleCallback || ((cb) => setTimeout(cb, 1));

const NORMALIZERS = {
  hourly:              (d) => Array.isArray(d) ? Object.fromEntries(d.map(r => [r.hour,           r.count])) : d,
  weekly:              (d) => Array.isArray(d) ? Object.fromEntries(d.map(r => [r.day,            r.count])) : d,
  severity:            (d) => Array.isArray(d) ? Object.fromEntries(d.map(r => [r.label,          r.count])) : d,
  weather:             (d) => Array.isArray(d) ? Object.fromEntries(d.map(r => [r.weather,        r.count])) : d,
  top_areas:           (d) => Array.isArray(d) ? Object.fromEntries(d.map(r => [r.area,           r.count])) : d,
  collision_types:     (d) => Array.isArray(d) ? Object.fromEntries(d.map(r => [r.collision_type, r.count])) : d,
  causes:              (d) => Array.isArray(d) ? Object.fromEntries(d.map(r => [r.cause,          r.count])) : d,
  severity_by_weather: (d) => d,
};

function Analytics() {
  const [data, setData] = useState({});
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    // Performance optimization: load critical charts first
    const criticalEndpoints = ['severity', 'hourly'];
    
    Promise.allSettled([
      ...criticalEndpoints.map(e => fetchEda(e).then(r => ({ key: e, data: r.data }))),
    ]).then(results => {
      const d = {};
      results.forEach(r => {
        if (r.status === 'fulfilled') {
          const norm = NORMALIZERS[r.value.key];
          d[r.value.key] = norm ? norm(r.value.data) : r.value.data;
        }
      });
      if (Object.keys(d).length) setData(d);
    });
    
    // Defer remaining charts (performance optimization - non-visual)
    requestIdleCallback(() => {
      const remainingEndpoints = ['weekly', 'weather', 'top_areas', 'collision_types', 'causes'];
      Promise.allSettled([
        ...remainingEndpoints.map(e => fetchEda(e).then(r => ({ key: e, data: r.data }))),
        fetchModelMetrics().then(r => r.data),
      ]).then(results => {
        setData(prev => {
          const d = { ...prev };
          results.slice(0, remainingEndpoints.length).forEach(r => {
            if (r.status === 'fulfilled') {
              const norm = NORMALIZERS[r.value.key];
              d[r.value.key] = norm ? norm(r.value.data) : r.value.data;
            }
          });
          return d;
        });
        const metricsResult = results[remainingEndpoints.length];
        if (metricsResult.status === 'fulfilled') setMetrics(metricsResult.value);
      });
    }, { timeout: 2000 });
  }, []);

  const toArr = useCallback((obj) => obj ? Object.entries(obj).map(([name, value]) => ({ name, value })) : [], []);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-bold" style={{ color: 'var(--clr-text)' }}>Analytics</h2>
        <p className="text-sm" style={{ color: 'var(--clr-text-muted)' }}>
          Exploratory Data Analysis & Model Performance
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

      {/* Charts grid */}
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
            <div className="flex flex-wrap gap-3 justify-center mt-2">
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

export default memo(Analytics);
