import { memo } from 'react';

const StatCard = memo(({ icon, label, value, sub, color = 'var(--clr-primary)' }) => {
  return (
    <div className="rounded-xl sm:rounded-2xl p-4 sm:p-5 lg:p-6 border transition-all duration-300 hover:translate-y-[-2px] hover:shadow-xl"
      style={{ 
        background: 'var(--clr-surface)', 
        borderColor: 'var(--clr-border)',
        boxShadow: '0 2px 8px rgba(0,0,0,0.04)'
      }}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-xs sm:text-xs font-semibold uppercase tracking-wider mb-2"
            style={{ color: 'var(--clr-text-muted)' }}>{label}</p>
          <p className="text-2xl sm:text-3xl lg:text-2xl font-bold truncate" style={{ color }}>{value}</p>
          {sub && <p className="text-xs sm:text-sm mt-1.5" style={{ color: 'var(--clr-text-muted)' }}>{sub}</p>}
        </div>
        {icon && (
          <div className="w-12 h-12 sm:w-14 sm:h-14 lg:w-12 lg:h-12 rounded-xl sm:rounded-2xl flex items-center justify-center shrink-0 shadow-lg"
            style={{ background: `${color}20`, color }}>
            {icon}
          </div>
        )}
      </div>
    </div>
  );
});

StatCard.displayName = 'StatCard';

export default StatCard;
