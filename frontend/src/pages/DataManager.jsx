import { useState, useRef, useEffect, memo } from 'react';
import {
  FiUploadCloud, FiPlay, FiCheckCircle, FiXCircle,
  FiTrash2, FiClock, FiDatabase, FiInfo, FiAlertTriangle,
} from 'react-icons/fi';
import { uploadCSV, runPipeline, fetchUploads, deleteUpload } from '../api';

// ─── CSV format reference ────────────────────────────────────────────────────
const REQUIRED_COLUMNS = [
  { col: 'Accident_severity',           type: 'Category', example: 'Slight Injury / Serious Injury / Fatal Injury', note: 'Target variable' },
  { col: 'Time',                        type: 'String',   example: '17:02:00', note: 'HH:MM:SS — hour extracted automatically' },
  { col: 'Day_of_week',                 type: 'Category', example: 'Monday … Sunday' },
  { col: 'Area_accident_occured',       type: 'Category', example: 'Office areas, Residential areas, Market areas, Rural village areas …', note: 'Geocoded to Indian cities' },
  { col: 'Weather_conditions',          type: 'Category', example: 'Normal, Raining, Fog or mist, Windy, Cloudy' },
  { col: 'Light_conditions',            type: 'Category', example: 'Daylight, Darkness - lights lit, Darkness - lights unlit' },
  { col: 'Road_surface_type',           type: 'Category', example: 'Asphalt roads, Gravel roads, Earth roads' },
  { col: 'Road_surface_conditions',     type: 'Category', example: 'Dry, Wet or damp, Snow, Flood over 3cm. deep' },
  { col: 'Type_of_collision',           type: 'Category', example: 'Vehicle with vehicle collision, Rollover, Collision with pedestrians' },
  { col: 'Type_of_vehicle',             type: 'Category', example: 'Automobile, Lorry (above 10t), Public (> 45 seats)' },
  { col: 'Cause_of_accident',           type: 'Category', example: 'No distancing, Driving carelessly, Overtaking' },
  { col: 'Road_allignment',             type: 'Category', example: 'Tangent road with flat terrain, Gentle horizontal curve' },
  { col: 'Types_of_Junction',           type: 'Category', example: 'No junction, Y Shape, T Shape, X Shape' },
  { col: 'Lanes_or_Medians',            type: 'Category', example: 'Undivided Two way, Two-way (divided with solid lines)' },
  { col: 'Number_of_vehicles_involved', type: 'Numeric',  example: '1, 2, 3 …' },
  { col: 'Driving_experience',          type: 'Category', example: 'Below 1yr, 1-2yr, 2-5yr, 5-10yr, Above 10yr' },
  { col: 'Age_band_of_driver',          type: 'Category', example: 'Under 18, 18-30, 31-50, Over 51' },
  { col: 'Vehicle_movement',            type: 'Category', example: 'Going straight, U-Turn, Turning left' },
  { col: 'Pedestrian_movement',         type: 'Category', example: 'Not a Pedestrian, Crossing from driver near side' },
];

// ─── helpers ─────────────────────────────────────────────────────────────────
function fmtBytes(b) {
  if (b == null) return '—';
  return b > 1_048_576 ? `${(b / 1_048_576).toFixed(1)} MB` : `${(b / 1024).toFixed(1)} KB`;
}
function fmtDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}

const STATUS_STYLE = {
  uploaded: { bg: '#3b82f615', color: '#3b82f6', label: 'Uploaded' },
  running:  { bg: '#f59e0b15', color: '#f59e0b', label: 'Running…' },
  complete: { bg: '#22c55e15', color: '#22c55e', label: 'Complete' },
  failed:   { bg: '#ef444415', color: '#ef4444', label: 'Failed'   },
};

// ─── component ───────────────────────────────────────────────────────────────
function DataManager() {
  const fileRef = useRef();
  const [uploadStatus, setUploadStatus]   = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [uploading, setUploading]         = useState(false);
  const [running, setRunning]             = useState(false);
  const [history, setHistory]             = useState([]);
  const [deleting, setDeleting]           = useState(null);   // entry id being deleted
  const [showFormat, setShowFormat]       = useState(false);

  const loadHistory = () =>
    fetchUploads().then(r => setHistory(r.data)).catch(() => {});

  useEffect(() => { loadHistory(); }, []);

  const handleUpload = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setUploading(true); setUploadStatus(null);
    try {
      const res = await uploadCSV(file);
      setUploadStatus({ ok: true, msg: `${res.data.filename} uploaded (${fmtBytes(res.data.size_bytes)})` });
      loadHistory();
    } catch (err) {
      setUploadStatus({ ok: false, msg: err.response?.data?.error || 'Upload failed' });
    } finally { setUploading(false); }
  };

  const handlePipeline = async () => {
    setRunning(true); setPipelineStatus(null);
    loadHistory();          // refresh to show "running" status
    try {
      const res = await runPipeline();
      setPipelineStatus({ ok: true, msg: res.data.message });
    } catch (err) {
      setPipelineStatus({ ok: false, msg: err.response?.data?.error || 'Pipeline failed' });
    } finally {
      setRunning(false);
      loadHistory();        // refresh with final result
    }
  };

  const handleDelete = async (id, clearArtifacts) => {
    setDeleting(id);
    try {
      await deleteUpload(id, clearArtifacts);
      loadHistory();
    } catch (err) {
      alert(err.response?.data?.error || 'Delete failed');
    } finally { setDeleting(null); }
  };

  return (
    <div className="space-y-4 sm:space-y-6 max-w-[1200px] mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-xl sm:text-2xl lg:text-3xl font-bold" style={{ color: 'var(--clr-text)' }}>Data Manager</h2>
        <p className="text-sm sm:text-base mt-1" style={{ color: 'var(--clr-text-muted)' }}>Upload datasets and manage the ML pipeline</p>
      </div>

      {/* ── Upload ─────────────────────────────────────────────────────────── */}
      <div className="rounded-xl sm:rounded-2xl border p-4 sm:p-6 space-y-4 shadow-lg"
        style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: 'var(--clr-primary)20', color: 'var(--clr-primary)' }}>
            <FiUploadCloud size={20} />
          </div>
          <div className="min-w-0">
            <h3 className="text-sm sm:text-base font-bold" style={{ color: 'var(--clr-text)' }}>Upload CSV Dataset</h3>
            <p className="text-xs sm:text-sm" style={{ color: 'var(--clr-text-muted)' }}>Select a road accident CSV file to upload</p>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-3">
          <input ref={fileRef} type="file" accept=".csv"
            className="flex-1 text-xs sm:text-sm file:mr-3 file:py-2 file:px-4 file:rounded-xl file:border-0 file:text-xs sm:file:text-sm file:font-medium file:cursor-pointer"
            style={{ color: 'var(--clr-text-muted)' }} />
          <button onClick={handleUpload} disabled={uploading}
            className="px-5 sm:px-6 py-2.5 sm:py-3 rounded-xl text-sm font-bold text-white shrink-0 shadow-lg"
            style={{ background: uploading ? 'var(--clr-border)' : 'var(--clr-primary)' }}>
            {uploading ? 'Uploading…' : 'Upload'}
          </button>
        </div>

        {uploadStatus && (
          <div className="flex items-center gap-2 text-sm" style={{ color: uploadStatus.ok ? '#22c55e' : '#ef4444' }}>
            {uploadStatus.ok ? <FiCheckCircle /> : <FiXCircle />}
            {uploadStatus.msg}
          </div>
        )}

        {/* CSV format guide toggle */}
        <button onClick={() => setShowFormat(v => !v)}
          className="flex items-center gap-2 text-xs sm:text-sm font-semibold mt-1 transition-colors"
          style={{ color: 'var(--clr-primary-light)' }}>
          <FiInfo size={14} />
          {showFormat ? 'Hide' : 'Show'} expected CSV format &amp; column reference
        </button>

        {showFormat && (
          <div className="rounded-xl border p-4 space-y-3 text-xs sm:text-sm"
            style={{ background: 'var(--clr-surface-2)', borderColor: 'var(--clr-border)' }}>

            {/* Summary */}
            <div className="space-y-1">
              <p className="font-bold" style={{ color: 'var(--clr-text)' }}>File requirements</p>
              <ul className="space-y-0.5 list-disc list-inside" style={{ color: 'var(--clr-text-muted)' }}>
                <li>Format: <strong>CSV</strong> with a header row</li>
                <li>Encoding: <strong>UTF-8</strong></li>
                <li>Minimum rows: <strong>~500</strong> (fewer will produce sparse clusters)</li>
                <li>Missing values are filled with <code className="px-1 rounded" style={{ background: 'var(--clr-border)' }}>Unknown</code> automatically</li>
              </ul>
            </div>

            {/* First CSV line example */}
            <div className="space-y-1">
              <p className="font-bold" style={{ color: 'var(--clr-text)' }}>Sample header row</p>
              <div className="rounded p-2 overflow-x-auto font-mono leading-relaxed text-xs"
                style={{ background: 'var(--clr-border)', color: 'var(--clr-text)' }}>
                Time,Day_of_week,Age_band_of_driver,Driving_experience,Type_of_vehicle,Area_accident_occured,
                Lanes_or_Medians,Road_allignment,Types_of_Junction,Road_surface_type,Road_surface_conditions,
                Light_conditions,Weather_conditions,Type_of_collision,Number_of_vehicles_involved,
                Vehicle_movement,Pedestrian_movement,Cause_of_accident,Accident_severity
              </div>
            </div>

            {/* Column table */}
            <div className="space-y-1">
              <p className="font-bold" style={{ color: 'var(--clr-text)' }}>Required columns ({REQUIRED_COLUMNS.length})</p>
              <div className="overflow-x-auto -mx-4 sm:mx-0">
                <div className="inline-block min-w-full align-middle px-4 sm:px-0">
                  <table className="w-full border-collapse text-xs">
                    <thead>
                      <tr style={{ color: 'var(--clr-text-muted)', borderBottom: '1px solid var(--clr-border)' }}>
                        <th className="text-left py-2 pr-3 font-semibold whitespace-nowrap">Column</th>
                        <th className="text-left py-2 pr-3 font-semibold">Type</th>
                        <th className="text-left py-2 pr-3 font-semibold">Example values</th>
                        <th className="text-left py-2 font-semibold">Note</th>
                      </tr>
                    </thead>
                    <tbody>
                      {REQUIRED_COLUMNS.map(({ col, type, example, note }) => (
                        <tr key={col} style={{ borderBottom: '1px solid var(--clr-border)' }}>
                          <td className="py-2 pr-3 font-mono whitespace-nowrap"
                            style={{ color: 'var(--clr-primary-light)' }}>{col}</td>
                          <td className="py-2 pr-3" style={{ color: 'var(--clr-text-muted)' }}>{type}</td>
                          <td className="py-2 pr-3" style={{ color: 'var(--clr-text)' }}>{example}</td>
                          <td className="py-2" style={{ color: 'var(--clr-text-muted)', fontStyle: note ? 'italic' : 'normal' }}>{note || ''}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Severity mapping */}
            <div className="flex flex-col sm:flex-row gap-4 sm:gap-6 pt-1">
              <div>
                <p className="font-bold mb-1" style={{ color: 'var(--clr-text)' }}>Accident_severity mapping</p>
                <div className="space-y-0.5" style={{ color: 'var(--clr-text-muted)' }}>
                  <p><span className="font-mono" style={{ color: '#22c55e' }}>Slight Injury</span>  →  1</p>
                  <p><span className="font-mono" style={{ color: '#f59e0b' }}>Serious Injury</span> →  2</p>
                  <p><span className="font-mono" style={{ color: '#ef4444' }}>Fatal Injury</span>   →  3</p>
                </div>
              </div>
              <div>
                <p className="font-bold mb-1" style={{ color: 'var(--clr-text)' }}>Weather → binned category</p>
                <div className="space-y-0.5" style={{ color: 'var(--clr-text-muted)' }}>
                  <p>Normal, Cloudy → Clear</p>
                  <p>Raining → Rain</p>
                  <p>Fog or mist → Fog</p>
                  <p>Windy → Wind</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Pipeline ───────────────────────────────────────────────────────── */}
      <div className="rounded-xl sm:rounded-2xl border p-4 sm:p-6 space-y-4 shadow-lg"
        style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl flex items-center justify-center shrink-0"
            style={{ background: '#22c55e20', color: '#22c55e' }}>
            <FiPlay size={20} />
          </div>
          <div className="min-w-0">
            <h3 className="text-sm sm:text-base font-bold" style={{ color: 'var(--clr-text)' }}>Run ML Pipeline</h3>
            <p className="text-xs sm:text-sm" style={{ color: 'var(--clr-text-muted)' }}>
              Preprocess → EDA → DBSCAN → Random Forest → ARI
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {['Preprocess', 'EDA', 'DBSCAN', 'Classifier', 'ARI'].map((step, i) => (
            <div key={step} className="flex items-center gap-2">
              <span className="px-3 py-1.5 rounded-xl text-xs font-semibold"
                style={{ background: 'var(--clr-surface-2)', color: 'var(--clr-text-muted)' }}>
                {step}
              </span>
              {i < 4 && <span style={{ color: 'var(--clr-border)' }}>→</span>}
            </div>
          ))}
        </div>

        <button onClick={handlePipeline} disabled={running}
          className="w-full py-3 sm:py-4 rounded-xl text-sm sm:text-base font-bold text-white flex items-center justify-center gap-2 shadow-lg"
          style={{ background: running ? 'var(--clr-border)' : '#22c55e' }}>
          <FiPlay size={18} />
          {running ? 'Running pipeline… (this may take a few minutes)' : 'Run Full Pipeline'}
        </button>

        {pipelineStatus && (
          <div className="flex items-center gap-2 text-sm" style={{ color: pipelineStatus.ok ? '#22c55e' : '#ef4444' }}>
            {pipelineStatus.ok ? <FiCheckCircle /> : <FiXCircle />}
            {pipelineStatus.msg}
          </div>
        )}
      </div>

      {/* ── Upload history ─────────────────────────────────────────────────── */}
      <div className="rounded-xl sm:rounded-2xl border shadow-lg" style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 px-4 sm:px-6 py-4 border-b"
          style={{ borderColor: 'var(--clr-border)' }}>
          <div className="flex items-center gap-2">
            <FiClock size={18} style={{ color: 'var(--clr-text-muted)' }} />
            <h3 className="text-sm sm:text-base font-bold" style={{ color: 'var(--clr-text)' }}>
              Upload History
            </h3>
            <span className="text-xs px-2.5 py-1 rounded-full font-semibold"
              style={{ background: 'var(--clr-surface-2)', color: 'var(--clr-text-muted)' }}>
              {history.length}
            </span>
          </div>
          {history.length > 0 && (
            <button onClick={loadHistory}
              className="text-xs sm:text-sm px-3 sm:px-4 py-2 rounded-xl font-semibold"
              style={{ background: 'var(--clr-surface-2)', color: 'var(--clr-text-muted)' }}>
              Refresh
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 sm:py-16 gap-3">
            <FiDatabase size={40} style={{ color: 'var(--clr-border)' }} />
            <p className="text-sm" style={{ color: 'var(--clr-text-muted)' }}>No uploads yet</p>
          </div>
        ) : (
          <div className="divide-y" style={{ borderColor: 'var(--clr-border)' }}>
            {history.map(entry => {
              const s = STATUS_STYLE[entry.status] || STATUS_STYLE.uploaded;
              const pr = entry.pipeline_result;
              return (
                <div key={entry.id} className="px-4 sm:px-6 py-4 sm:py-5 space-y-3">
                  {/* Row header */}
                  <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                      <FiDatabase size={18} style={{ color: 'var(--clr-text-muted)', flexShrink: 0 }} />
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-bold truncate" style={{ color: 'var(--clr-text)' }}>
                          {entry.filename}
                        </p>
                        <p className="text-xs sm:text-sm" style={{ color: 'var(--clr-text-muted)' }}>
                          {fmtDate(entry.uploaded_at)} · {fmtBytes(entry.size_bytes)}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 shrink-0 flex-wrap">
                      {/* Status badge */}
                      <span className="text-xs font-bold px-3 py-1.5 rounded-full"
                        style={{ background: s.bg, color: s.color }}>
                        {s.label}
                      </span>

                      {/* Delete (entry only) */}
                      <button
                        title="Remove from history"
                        disabled={deleting === entry.id}
                        onClick={() => handleDelete(entry.id, false)}
                        className="p-2 rounded-xl transition-colors"
                        style={{ color: 'var(--clr-text-muted)', background: 'transparent' }}
                        onMouseEnter={e => e.currentTarget.style.background = '#ef444420'}
                        onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                        <FiXCircle size={16} />
                      </button>

                      {/* Delete + clear artifacts */}
                      <button
                        title="Delete entry AND clear all generated models/EDA files"
                        disabled={deleting === entry.id}
                        onClick={() => {
                          if (window.confirm('Delete this entry and clear all generated models, EDA files, and the uploaded CSV?\n\nThe dashboard will go back to an empty state.'))
                            handleDelete(entry.id, true);
                        }}
                        className="p-2 rounded-xl flex items-center gap-1.5 text-xs font-bold transition-colors"
                        style={{ color: '#ef4444', background: '#ef444410' }}
                        onMouseEnter={e => e.currentTarget.style.background = '#ef444425'}
                        onMouseLeave={e => e.currentTarget.style.background = '#ef444410'}>
                        <FiTrash2 size={14} /> <span className="hidden sm:inline">Clear data</span>
                      </button>
                    </div>
                  </div>

                  {/* Pipeline result metrics */}
                  {entry.status === 'complete' && pr && (
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                      {[
                        { label: 'Records',   value: pr.records?.toLocaleString() ?? '—' },
                        { label: 'Clusters',  value: pr.clusters ?? '—' },
                        { label: 'Accuracy',  value: pr.accuracy != null ? `${(pr.accuracy * 100).toFixed(1)}%` : '—' },
                        { label: 'ARI Range', value: pr.ari_range ? `${pr.ari_range[0]} – ${pr.ari_range[1]}` : '—' },
                      ].map(({ label, value }) => (
                        <div key={label} className="rounded-xl px-3 sm:px-4 py-3"
                          style={{ background: 'var(--clr-surface-2)' }}>
                          <p className="text-xs mb-1" style={{ color: 'var(--clr-text-muted)' }}>{label}</p>
                          <p className="text-sm font-bold font-mono" style={{ color: 'var(--clr-text)' }}>{value}</p>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Error message */}
                  {entry.status === 'failed' && pr?.error && (
                    <div className="flex items-start gap-2 rounded-xl p-3 text-xs"
                      style={{ background: '#ef444410', color: '#ef4444' }}>
                      <FiAlertTriangle size={14} className="mt-0.5 shrink-0" />
                      <span className="font-mono break-all">{pr.error}</span>
                    </div>
                  )}

                  {/* Completed at */}
                  {entry.status === 'complete' && pr?.completed_at && (
                    <p className="text-xs" style={{ color: 'var(--clr-text-muted)' }}>
                      Pipeline completed · {fmtDate(pr.completed_at)}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default memo(DataManager);
