import { Routes, Route, NavLink, useLocation } from 'react-router-dom';
import { FiMap, FiBarChart2, FiActivity, FiUploadCloud, FiCpu, FiSun, FiMoon } from 'react-icons/fi';
import { useTheme } from './ThemeContext';
import Dashboard from './pages/Dashboard';
import MapView from './pages/MapView';
import Analytics from './pages/Analytics';
import Predict from './pages/Predict';
import DataManager from './pages/DataManager';

const navItems = [
  { to: '/', icon: <FiActivity size={18} />, label: 'Dashboard' },
  { to: '/map', icon: <FiMap size={18} />, label: 'Hotspot Map' },
  { to: '/analytics', icon: <FiBarChart2 size={18} />, label: 'Analytics' },
  { to: '/predict', icon: <FiCpu size={18} />, label: 'Predict' },
  { to: '/data', icon: <FiUploadCloud size={18} />, label: 'Data' },
];

export default function App() {
  const location = useLocation();
  const { theme, toggle } = useTheme();

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 shrink-0 flex flex-col border-r transition-colors duration-250"
        style={{ background: 'var(--clr-surface)', borderColor: 'var(--clr-border)' }}>
        <div className="p-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg flex items-center justify-center text-white font-bold text-sm"
              style={{ background: 'linear-gradient(135deg, var(--clr-primary), var(--clr-danger))' }}>
              AH
            </div>
            <div>
              <h1 className="text-sm font-semibold" style={{ color: 'var(--clr-text)' }}>AcciHotspot</h1>
              <p className="text-xs" style={{ color: 'var(--clr-text-muted)' }}>AI Prediction System</p>
            </div>
          </div>
          <button onClick={toggle} aria-label="Toggle theme"
            className="w-8 h-8 rounded-lg flex items-center justify-center transition-colors duration-200"
            style={{ background: 'var(--clr-surface-2)', color: 'var(--clr-text-muted)' }}>
            {theme === 'dark' ? <FiSun size={15} /> : <FiMoon size={15} />}
          </button>
        </div>

        <nav className="flex-1 px-3 mt-2 space-y-1">
          {navItems.map(({ to, icon, label }) => {
            const active = location.pathname === to;
            return (
              <NavLink key={to} to={to}
                className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200"
                style={{
                  background: active ? 'rgba(99,102,241,.15)' : 'transparent',
                  color: active ? 'var(--clr-primary-light)' : 'var(--clr-text-muted)',
                }}>
                {icon}
                {label}
              </NavLink>
            );
          })}
        </nav>

        <div className="p-4 m-3 rounded-xl text-xs" style={{ background: 'var(--clr-surface-2)' }}>
          <p style={{ color: 'var(--clr-text-muted)' }}>Powered by</p>
          <p className="font-medium mt-0.5" style={{ color: 'var(--clr-text)' }}>DBSCAN + Random Forest</p>
          <p style={{ color: 'var(--clr-text-muted)' }}>GIS & Machine Learning</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-6 transition-colors duration-250" style={{ background: 'var(--clr-bg)' }}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/map" element={<MapView />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/data" element={<DataManager />} />
        </Routes>
      </main>
    </div>
  );
}
