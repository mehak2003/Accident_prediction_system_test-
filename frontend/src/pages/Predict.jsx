import { useState, useEffect, memo } from 'react';
import { FiCpu, FiAlertTriangle } from 'react-icons/fi';
import RiskBadge from '../components/RiskBadge';
import { predictSeverity, fetchClusters } from '../api';

const WEATHER_OPTIONS = ['Clear', 'Rain', 'Fog', 'Snow', 'Wind', 'Other'];
const LIGHT_OPTIONS = ['Daylight', 'Darkness - lights lit', 'Darkness - lights unlit', 'Darkness - no lighting'];
const SURFACE_COND = ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep'];
const COLLISION_TYPES = ['Vehicle with vehicle collision', 'Collision with roadside objects', 'Collision with pedestrians', 'Rollover', 'Collision with animals', 'Other'];
const CAUSES = ['No distancing', 'Changing lane to the right', 'Changing lane to the left', 'Driving carelessly', 'No priority to vehicle', 'Moving Backward', 'Overtaking', 'Driving under the influence of drugs', 'Driving at high speed', 'Other'];
const EXPERIENCE = ['Below 1yr', '1-2yr', '2-5yr', '5-10yr', 'Above 10yr', 'unknown'];
const AGE_BANDS = ['Under 18', '18-30', '31-50', 'Over 51', 'Unknown'];

const defaultForm = {
  cluster_id: 0, hour: 12, day_of_week: 3, is_night: 0,
  weather: 'Clear', num_vehicles: 2,
  light_conditions: 'Daylight', road_surface_conditions: 'Dry',
  type_of_collision: 'Vehicle with vehicle collision',
  cause_of_accident: 'No distancing',
  driving_experience: '5-10yr', age_band_of_driver: '18-30',
};

const SelectField = memo(({ label, name, value, options, onChange }) => {
  return (
    <div>
      <label className="block text-xs sm:text-sm font-semibold mb-2" style={{ color: 'var(--clr-text-muted)' }}>{label}</label>
      <select name={name} value={value} onChange={onChange}
        className="w-full px-3 sm:px-4 py-2.5 sm:py-3 rounded-xl text-sm border outline-none focus:ring-2 transition-all"
        style={{ background: 'var(--clr-surface-2)', borderColor: 'var(--clr-border)', color: 'var(--clr-text)', '--tw-ring-color': 'var(--clr-primary)' }}>
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    </div>
  );
});

SelectField.displayName = 'SelectField';

const NumberField = memo(({ label, name, value, min, max, onChange }) => {
  return (
    <div>
      <label className="block text-xs sm:text-sm font-semibold mb-2" style={{ color: 'var(--clr-text-muted)' }}>{label}</label>
      <input type="number" name={name} value={value} min={min} max={max} onChange={onChange}
        className="w-full px-3 sm:px-4 py-2.5 sm:py-3 rounded-xl text-sm border outline-none focus:ring-2 transition-all"
        style={{ background: 'var(--clr-surface-2)', borderColor: 'var(--clr-border)', color: 'var(--clr-text)', '--tw-ring-color': 'var(--clr-primary)' }} />
    </div>
  );
});

NumberField.displayName = 'NumberField';

function Predict() {
  const [form, setForm] = useState(defaultForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [clusterIds, setClusterIds] = useState([]);

  useEffect(() => {
    fetchClusters()
      .then(r => {
        const ids = (r.data?.features || [])
          .map(f => f.properties?.Cluster_ID)
          .filter(id => id !== undefined && id !== null)
          .sort((a, b) => a - b);
        if (ids.length) {
          setClusterIds(ids);
          setForm(prev => ({ ...prev, cluster_id: ids[0] }));
        }
      })
      .catch(() => {});
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true); setError(null); setResult(null);
    try {
      const payload = { ...form, cluster_id: +form.cluster_id, hour: +form.hour, day_of_week: +form.day_of_week, is_night: +form.is_night, num_vehicles: +form.num_vehicles };
      const res = await predictSeverity(payload);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed');
    } finally { setLoading(false); }
  };

  return (
    <div className="space-y-4 sm:space-y-6 lg:space-y-8 max-w-[1400px] mx-auto">
      {/* Header */}
      <div>
        <h2 className="text-xl sm:text-2xl lg:text-3xl font-bold" style={{ color: 'var(--clr-text)' }}>Risk Prediction</h2>
        <p className="text-sm sm:text-base mt-1" style={{ color: 'var(--clr-text-muted)' }}>
          Predict accident severity and ARI for specific conditions
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 sm:gap-6">
        {/* Form */}
        <form onSubmit={handleSubmit} className="lg:col-span-3 rounded-xl sm:rounded-2xl border p-4 sm:p-6 space-y-5 shadow-lg"
          style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
            {clusterIds.length > 0 ? (
              <SelectField label="Cluster ID" name="cluster_id" value={form.cluster_id} options={clusterIds} onChange={handleChange} />
            ) : (
              <NumberField label="Cluster ID" name="cluster_id" value={form.cluster_id} min={0} max={999} onChange={handleChange} />
            )}
            <NumberField label="Hour (0-23)" name="hour" value={form.hour} min={0} max={23} onChange={handleChange} />
            <NumberField label="Day of Week (0-6)" name="day_of_week" value={form.day_of_week} min={0} max={6} onChange={handleChange} />
            <SelectField label="Night?" name="is_night" value={form.is_night} options={[0, 1]} onChange={handleChange} />
            <NumberField label="Num Vehicles" name="num_vehicles" value={form.num_vehicles} min={1} max={10} onChange={handleChange} />
            <SelectField label="Weather" name="weather" value={form.weather} options={WEATHER_OPTIONS} onChange={handleChange} />
            <SelectField label="Light Conditions" name="light_conditions" value={form.light_conditions} options={LIGHT_OPTIONS} onChange={handleChange} />
            <SelectField label="Road Surface" name="road_surface_conditions" value={form.road_surface_conditions} options={SURFACE_COND} onChange={handleChange} />
            <SelectField label="Collision Type" name="type_of_collision" value={form.type_of_collision} options={COLLISION_TYPES} onChange={handleChange} />
            <SelectField label="Cause" name="cause_of_accident" value={form.cause_of_accident} options={CAUSES} onChange={handleChange} />
            <SelectField label="Driving Experience" name="driving_experience" value={form.driving_experience} options={EXPERIENCE} onChange={handleChange} />
            <SelectField label="Driver Age Band" name="age_band_of_driver" value={form.age_band_of_driver} options={AGE_BANDS} onChange={handleChange} />
          </div>

          <button type="submit" disabled={loading}
            className="w-full flex items-center justify-center gap-2 py-3 sm:py-4 rounded-xl text-sm sm:text-base font-bold text-white shadow-lg"
            style={{ background: loading ? 'var(--clr-border)' : 'var(--clr-primary)' }}>
            <FiCpu size={18} />
            {loading ? 'Predicting...' : 'Run Prediction'}
          </button>
        </form>

        {/* Result */}
        <div className="lg:col-span-2 space-y-4">
          {error && (
            <div className="rounded-xl sm:rounded-2xl border p-4 sm:p-5 flex items-center gap-3 shadow-lg"
              style={{ background: '#ef444415', borderColor: '#ef444440', color: '#ef4444' }}>
              <FiAlertTriangle size={20} className="shrink-0" />
              <p className="text-sm">{error}</p>
            </div>
          )}

          {result && (
            <>
              <div className="rounded-xl sm:rounded-2xl border p-5 sm:p-6 text-center shadow-lg"
                style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
                <p className="text-xs sm:text-sm uppercase tracking-wider mb-3" style={{ color: 'var(--clr-text-muted)' }}>
                  Predicted Severity
                </p>
                <p className="text-3xl sm:text-4xl font-bold mb-3" style={{ color: 'var(--clr-primary-light)' }}>
                  {result.predicted_label}
                </p>
                <RiskBadge tier={result.risk_tier} />
              </div>

              <div className="rounded-xl sm:rounded-2xl border p-5 sm:p-6 shadow-lg"
                style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
                <p className="text-xs sm:text-sm uppercase tracking-wider mb-3" style={{ color: 'var(--clr-text-muted)' }}>
                  ARI Score
                </p>
                <div className="relative h-4 sm:h-5 rounded-full overflow-hidden mb-3" style={{ background: 'var(--clr-surface-2)' }}>
                  <div className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
                    style={{
                      width: `${(result.ari_score || 0) * 100}%`,
                      background: result.risk_tier === 'Critical' ? '#ef4444'
                        : result.risk_tier === 'Severe' ? '#f97316'
                        : result.risk_tier === 'Moderate' ? '#f59e0b' : '#22c55e',
                    }} />
                </div>
                <p className="text-2xl sm:text-3xl font-bold font-mono" style={{ color: 'var(--clr-text)' }}>
                  {result.ari_score}
                </p>
              </div>

              <div className="rounded-xl sm:rounded-2xl border p-5 sm:p-6 shadow-lg"
                style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
                <p className="text-xs sm:text-sm uppercase tracking-wider mb-4" style={{ color: 'var(--clr-text-muted)' }}>
                  Severity Probabilities
                </p>
                <div className="space-y-3">
                  {Object.entries(result.severity_probabilities || {}).map(([label, prob]) => (
                    <div key={label} className="flex items-center gap-3">
                      <span className="text-xs sm:text-sm w-20 sm:w-24 shrink-0 font-medium" style={{ color: 'var(--clr-text-muted)' }}>{label}</span>
                      <div className="flex-1 h-2.5 sm:h-3 rounded-full overflow-hidden" style={{ background: 'var(--clr-surface-2)' }}>
                        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${prob * 100}%`, background: 'var(--clr-primary)' }} />
                      </div>
                      <span className="text-xs sm:text-sm font-mono w-12 sm:w-14 text-right font-bold" style={{ color: 'var(--clr-text)' }}>
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="rounded-xl sm:rounded-2xl border p-5 sm:p-6 shadow-lg"
                style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
                <p className="text-xs sm:text-sm uppercase tracking-wider mb-4" style={{ color: 'var(--clr-text-muted)' }}>
                  ARI Weights
                </p>
                <div className="grid grid-cols-3 gap-3 sm:gap-4 text-center">
                  {Object.entries(result.weights || {}).map(([k, v]) => (
                    <div key={k}>
                      <p className="text-lg sm:text-xl font-bold font-mono" style={{ color: 'var(--clr-primary-light)' }}>{v}</p>
                      <p className="text-xs mt-1" style={{ color: 'var(--clr-text-muted)' }}>{k.replace('_', ' ')}</p>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {!result && !error && (
            <div className="rounded-xl sm:rounded-2xl border p-8 sm:p-10 text-center shadow-lg"
              style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
              <FiCpu size={40} className="mx-auto mb-3 opacity-30" style={{ color: 'var(--clr-border)' }} />
              <p className="text-sm" style={{ color: 'var(--clr-text-muted)' }}>
                Fill the form and run prediction
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default memo(Predict);
