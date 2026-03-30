import { useEffect, useState } from 'react';
import { FiAlertTriangle, FiMapPin, FiShield, FiTrendingUp } from 'react-icons/fi';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import StatCard from '../components/StatCard';
import ChartCard from '../components/ChartCard';
import RiskBadge from '../components/RiskBadge';
import { fetchClusters, fetchEdaSummary, fetchHealth } from '../api';
import { mockClusters, mockSummary, mockHealth } from '../mockData';

const PIE_COLORS = ['#22c55e', '#f59e0b', '#f97316', '#ef4444'];

export default function Dashboard() {
  const [clusters, setClusters] = useState(mockClusters);
  const [summary, setSummary] = useState(mockSummary);
  const [health, setHealth] = useState(mockHealth);
  const [live, setLive] = useState(false);

  useEffect(() => {
    Promise.allSettled([
      fetchClusters().then(r => r.data),
      fetchEdaSummary().then(r => r.data),
      fetchHealth().then(r => r.data),
    ]).then(([c, s, h]) => {
      if (c.status === 'fulfilled') { setClusters(c.value); setLive(true); }
      if (s.status === 'fulfilled') setSummary(s.value);
      if (h.status === 'fulfilled') setHealth(h.value);
    });
  }, []);

  const features = clusters?.features || [];
  const totalClusters = features.length;
  const criticalCount = features.filter(f => f.properties?.Risk_Tier === 'Critical').length;
  const totalIncidents = features.reduce((s, f) => s + (f.properties?.Incident_Count || 0), 0);
  const avgAri = totalClusters
    ? (features.reduce((s, f) => s + (f.properties?.ARI_Score || 0), 0) / totalClusters).toFixed(3)
    : '—';

  // Risk tier distribution for pie chart
  const tierCounts = {};
  features.forEach(f => {
    const t = f.properties?.Risk_Tier || 'Low';
    tierCounts[t] = (tierCounts[t] || 0) + 1;
  });
  const pieData = ['Low', 'Moderate', 'Severe', 'Critical']
    .filter(t => tierCounts[t])
    .map((t, i) => ({ name: t, value: tierCounts[t] }));

  // Top 5 clusters by ARI
  const topClusters = [...features]
    .sort((a, b) => (b.properties?.ARI_Score || 0) - (a.properties?.ARI_Score || 0))
    .slice(0, 6)
    .map(f => ({
      name: `C-${f.properties?.Cluster_ID}`,
      ari: +(f.properties?.ARI_Score || 0).toFixed(3),
      tier: f.properties?.Risk_Tier,
    }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold" style={{ color: 'var(--clr-text)' }}>Dashboard</h2>
          <p className="text-sm" style={{ color: 'var(--clr-text-muted)' }}>
            AI-Based Accident Hotspot Prediction Overview
            {!live && <span className="ml-2 text-xs px-2 py-0.5 rounded-full" style={{ background: '#f59e0b20', color: '#f59e0b' }}>Sample Data</span>}
          </p>
        </div>
        {health && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium"
            style={{
              background: health.status === 'ok' ? '#22c55e15' : '#ef444415',
              color: health.status === 'ok' ? '#22c55e' : '#ef4444',
            }}>
            <span className="w-2 h-2 rounded-full" style={{
              background: health.status === 'ok' ? '#22c55e' : '#ef4444'
            }} />
            System {health.status === 'ok' ? 'Online' : 'Offline'}
          </div>
        )}
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={<FiMapPin size={20} />} label="Total Clusters" value={totalClusters}
          sub="DBSCAN identified" color="var(--clr-info)" />
        <StatCard icon={<FiAlertTriangle size={20} />} label="Critical Zones" value={criticalCount}
          sub="Immediate attention" color="var(--clr-danger)" />
        <StatCard icon={<FiTrendingUp size={20} />} label="Total Incidents" value={totalIncidents.toLocaleString()}
          sub="In all clusters" color="var(--clr-warning)" />
        <StatCard icon={<FiShield size={20} />} label="Avg ARI Score" value={avgAri}
          sub="Across all clusters" color="var(--clr-primary)" />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <ChartCard title="Top Clusters by ARI Score" className="lg:col-span-2">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={topClusters}>
              <XAxis dataKey="name" tick={{ fill: 'var(--clr-text-muted)', fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: 'var(--clr-text-muted)', fontSize: 12 }} axisLine={false} tickLine={false} domain={[0, 1]} />
              <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)', borderRadius: 8, color: 'var(--clr-text)' }}
                labelStyle={{ color: 'var(--clr-text)' }} itemStyle={{ color: 'var(--clr-primary-light)' }} />
              <Bar dataKey="ari" radius={[6, 6, 0, 0]} fill="#6366f1" />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Risk Distribution">
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={55} outerRadius={85}
                paddingAngle={4} dataKey="value" stroke="none">
                {pieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
              </Pie>
              <Tooltip contentStyle={{ background: 'var(--clr-surface)', border: '1px solid var(--clr-border)', borderRadius: 8, color: 'var(--clr-text)' }}
                labelStyle={{ color: 'var(--clr-text)' }} />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-3 justify-center mt-2">
            {pieData.map((d, i) => (
              <div key={d.name} className="flex items-center gap-1.5 text-xs">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: PIE_COLORS[i] }} />
                <span style={{ color: 'var(--clr-text-muted)' }}>{d.name} ({d.value})</span>
              </div>
            ))}
          </div>
        </ChartCard>
      </div>

      {/* Cluster table */}
      <ChartCard title="Cluster Overview">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr style={{ color: 'var(--clr-text-muted)' }}>
                <th className="text-left py-2 px-3 font-medium">Cluster</th>
                <th className="text-left py-2 px-3 font-medium">Incidents</th>
                <th className="text-left py-2 px-3 font-medium">ARI Score</th>
                <th className="text-left py-2 px-3 font-medium">Risk Tier</th>
                <th className="text-left py-2 px-3 font-medium">Location</th>
              </tr>
            </thead>
            <tbody>
              {features.slice(0, 10).map((f, i) => {
                const p = f.properties || {};
                const [lon, lat] = f.geometry?.coordinates || [0, 0];
                return (
                  <tr key={i} className="border-t" style={{ borderColor: 'var(--clr-border)' }}>
                    <td className="py-2.5 px-3 font-medium">C-{p.Cluster_ID}</td>
                    <td className="py-2.5 px-3">{p.Incident_Count}</td>
                    <td className="py-2.5 px-3 font-mono">{(p.ARI_Score || 0).toFixed(3)}</td>
                    <td className="py-2.5 px-3"><RiskBadge tier={p.Risk_Tier} /></td>
                    <td className="py-2.5 px-3" style={{ color: 'var(--clr-text-muted)' }}>
                      {lat?.toFixed(4)}, {lon?.toFixed(4)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </ChartCard>
    </div>
  );
}
