import { useEffect, useState, useMemo, memo, useCallback, useRef } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import RiskBadge from '../components/RiskBadge';
import { fetchClusters } from '../api';
import { useTheme } from '../ThemeContext';

const tierColor = { Low: '#22c55e', Moderate: '#f59e0b', Severe: '#f97316', Critical: '#ef4444' };
const TIERS = ['Low', 'Moderate', 'Severe', 'Critical'];

// Performance optimization: memoized FitBounds component
const FitBounds = memo(({ features }) => {
  const map = useMap();
  const prevFeaturesLength = useRef(0);
  
  useEffect(() => {
    if (!features.length || features.length === prevFeaturesLength.current) return;
    prevFeaturesLength.current = features.length;
    
    const coords = features.map(f => {
      const [lon, lat] = f.geometry.coordinates;
      return [lat, lon];
    });
    
    requestAnimationFrame(() => {
      map.fitBounds(coords, { padding: [40, 40], animate: false });
    });
  }, [features, map]);
  
  return null;
});

FitBounds.displayName = 'FitBounds';

// Performance optimization: memoized ClusterMarker component
const ClusterMarker = memo(({ feature }) => {
  const p = feature.properties || {};
  const [lon, lat] = feature.geometry?.coordinates || [0, 0];
  const color = tierColor[p.Risk_Tier] || '#6366f1';
  const radius = Math.max(6, Math.min(20, (p.Incident_Count || 1) / 5));
  
  const pathOptions = useMemo(() => ({
    color,
    fillColor: color,
    fillOpacity: 0.35,
    weight: 2
  }), [color]);
  
  const center = useMemo(() => [lat, lon], [lat, lon]);
  
  return (
    <CircleMarker 
      center={center}
      radius={radius}
      pathOptions={pathOptions}
    >
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
              <p style={{ color: 'var(--clr-text-muted)' }}>Avg Severity</p>
              <p className="font-semibold">{p.Pred_Severity ? (+p.Pred_Severity).toFixed(2) : '—'}</p>
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
}, (prevProps, nextProps) => {
  return prevProps.feature.properties?.Cluster_ID === nextProps.feature.properties?.Cluster_ID &&
         prevProps.feature.properties?.Risk_Tier === nextProps.feature.properties?.Risk_Tier;
});

ClusterMarker.displayName = 'ClusterMarker';

function MapView() {
  const [clusters, setClusters] = useState(null);
  const [filter, setFilter] = useState('All');
  const { theme } = useTheme();
  const dataFetchedRef = useRef(false);

  const tileUrl = useMemo(() => 
    theme === 'dark'
      ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
      : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    [theme]
  );

  useEffect(() => {
    if (dataFetchedRef.current) return;
    dataFetchedRef.current = true;
    
    fetchClusters()
      .then(r => setClusters(r.data))
      .catch(() => {});
  }, []);

  const features = useMemo(() => clusters?.features || [], [clusters]);
  
  const filtered = useMemo(() => 
    filter === 'All' ? features : features.filter(f => f.properties?.Risk_Tier === filter),
    [features, filter]
  );

  const handleFilterChange = useCallback((tier) => {
    setFilter(tier);
  }, []);

  return (
    <div className="space-y-3 md:space-y-4 h-full flex flex-col">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-lg md:text-xl font-bold" style={{ color: 'var(--clr-text)' }}>Hotspot Map</h2>
          <p className="text-xs md:text-sm" style={{ color: 'var(--clr-text-muted)' }}>
            {filtered.length} clusters displayed
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {['All', ...TIERS].map(t => (
            <button key={t} onClick={() => handleFilterChange(t)}
              className="px-2.5 sm:px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
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

      <div className="flex-1 rounded-xl overflow-hidden border w-full" 
        style={{ borderColor: 'var(--clr-border)', minHeight: '400px', height: '100%' }}>
        <MapContainer 
          center={[20.5937, 78.9629]} 
          zoom={5} 
          className="h-full w-full"
          style={{ height: '100%', minHeight: '400px' }}
          preferCanvas={true}
          zoomControl={true}
          scrollWheelZoom={true}
          doubleClickZoom={true}
          touchZoom={true}
        >
          <TileLayer
            attribution='&copy; <a href="https://carto.com/">CARTO</a>'
            url={tileUrl}
            key={theme}
            maxZoom={18}
            minZoom={3}
            updateWhenIdle={true}
            updateWhenZooming={false}
            keepBuffer={2}
          />
          <FitBounds features={filtered} />
          {filtered.map((f) => (
            <ClusterMarker key={f.properties?.Cluster_ID || `cluster-${f.geometry?.coordinates?.join('-')}`} feature={f} />
          ))}
        </MapContainer>
      </div>

      <div className="flex flex-wrap items-center gap-3 sm:gap-4 px-3 sm:px-4 py-3 rounded-xl border"
        style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
        <span className="text-xs font-medium whitespace-nowrap" style={{ color: 'var(--clr-text-muted)' }}>Risk Levels:</span>
        {TIERS.map(t => (
          <div key={t} className="flex items-center gap-1.5 sm:gap-2 text-xs">
            <span className="w-3 h-3 rounded-full shrink-0" style={{ background: tierColor[t] }} />
            <span className="whitespace-nowrap" style={{ color: 'var(--clr-text-muted)' }}>{t}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default memo(MapView);
