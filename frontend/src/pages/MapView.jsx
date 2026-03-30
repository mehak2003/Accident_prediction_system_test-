import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import RiskBadge from '../components/RiskBadge';
import { fetchClusters } from '../api';
import { mockClusters } from '../mockData';
import { useTheme } from '../ThemeContext';

const tierColor = { Low: '#22c55e', Moderate: '#f59e0b', Severe: '#f97316', Critical: '#ef4444' };
const TIERS = ['Low', 'Moderate', 'Severe', 'Critical'];

function FitBounds({ features }) {
  const map = useMap();
  useEffect(() => {
    if (!features.length) return;
    const coords = features.map(f => {
      const [lon, lat] = f.geometry.coordinates;
      return [lat, lon];
    });
    map.fitBounds(coords, { padding: [40, 40] });
  }, [features, map]);
  return null;
}

export default function MapView() {
  const [clusters, setClusters] = useState(mockClusters);
  const [filter, setFilter] = useState('All');
  const [live, setLive] = useState(false);
  const { theme } = useTheme();

  const tileUrl = theme === 'dark'
    ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
    : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png';

  useEffect(() => {
    fetchClusters().then(r => { setClusters(r.data); setLive(true); }).catch(() => {});
  }, []);

  const features = clusters?.features || [];
  const filtered = filter === 'All' ? features : features.filter(f => f.properties?.Risk_Tier === filter);

  return (
    <div className="space-y-4 h-full flex flex-col">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold" style={{ color: 'var(--clr-text)' }}>Hotspot Map</h2>
          <p className="text-sm" style={{ color: 'var(--clr-text-muted)' }}>
            {filtered.length} clusters displayed
            {!live && <span className="ml-2 text-xs px-2 py-0.5 rounded-full" style={{ background: '#f59e0b20', color: '#f59e0b' }}>Sample Data</span>}
          </p>
        </div>
        <div className="flex gap-2">
          {['All', ...TIERS].map(t => (
            <button key={t} onClick={() => setFilter(t)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
              style={{
                background: filter === t ? 'var(--clr-primary)' : 'var(--clr-surface)',
                color: filter === t ? '#fff' : 'var(--clr-text-muted)',
                border: `1px solid ${filter === t ? 'var(--clr-primary)' : 'var(--clr-border)'}`,
              }}>
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 rounded-xl overflow-hidden border" style={{ borderColor: 'var(--clr-border)', minHeight: 500 }}>
        <MapContainer center={[20.5937, 78.9629]} zoom={5} className="h-full w-full"
          style={{ height: '100%', minHeight: 500 }}>
          <TileLayer
            attribution='&copy; <a href="https://carto.com/">CARTO</a>'
            url={tileUrl}
            key={theme}
          />
          <FitBounds features={filtered} />
          {filtered.map((f, i) => {
            const p = f.properties || {};
            const [lon, lat] = f.geometry?.coordinates || [0, 0];
            const color = tierColor[p.Risk_Tier] || '#6366f1';
            const radius = Math.max(6, Math.min(20, (p.Incident_Count || 1) / 5));
            return (
              <CircleMarker key={i} center={[lat, lon]} radius={radius}
                pathOptions={{ color, fillColor: color, fillOpacity: 0.35, weight: 2 }}>
                <Popup>
                  <div className="space-y-2 min-w-[180px]">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-sm">Cluster {p.Cluster_ID}</span>
                      <RiskBadge tier={p.Risk_Tier} />
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <p style={{ color: 'var(--clr-text-muted)' }}>Incidents</p>
                        <p className="font-semibold">{p.Incident_Count}</p>
                      </div>
                      <div>
                        <p style={{ color: 'var(--clr-text-muted)' }}>ARI Score</p>
                        <p className="font-semibold">{(p.ARI_Score || 0).toFixed(3)}</p>
                      </div>
                      <div>
                        <p style={{ color: 'var(--clr-text-muted)' }}>Severity</p>
                        <p className="font-semibold">{p.Pred_Severity || '—'}</p>
                      </div>
                      <div>
                        <p style={{ color: 'var(--clr-text-muted)' }}>Environment</p>
                        <p className="font-semibold">{p.Env_Modifier || '—'}</p>
                      </div>
                    </div>
                    <p className="text-xs" style={{ color: 'var(--clr-text-muted)' }}>
                      {lat.toFixed(4)}°N, {lon.toFixed(4)}°E
                    </p>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 px-4 py-3 rounded-xl border"
        style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
        <span className="text-xs font-medium" style={{ color: 'var(--clr-text-muted)' }}>Risk Levels:</span>
        {TIERS.map(t => (
          <div key={t} className="flex items-center gap-1.5 text-xs">
            <span className="w-3 h-3 rounded-full" style={{ background: tierColor[t] }} />
            <span style={{ color: 'var(--clr-text-muted)' }}>{t}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
