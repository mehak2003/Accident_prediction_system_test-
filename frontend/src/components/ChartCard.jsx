import { memo } from 'react';

const ChartCard = memo(({ title, children, className = '' }) => {
  return (
    <div className={`rounded-xl sm:rounded-2xl border p-4 sm:p-5 lg:p-6 transition-all duration-300 hover:shadow-lg ${className}`}
      style={{ 
        background: 'var(--clr-surface)', 
        borderColor: 'var(--clr-border)',
        boxShadow: '0 2px 8px rgba(0,0,0,0.04)'
      }}>
      {title && (
        <h3 className="text-sm sm:text-base font-bold mb-4 sm:mb-5" style={{ color: 'var(--clr-text)' }}>{title}</h3>
      )}
      {children}
    </div>
  );
});

ChartCard.displayName = 'ChartCard';

export default ChartCard;
