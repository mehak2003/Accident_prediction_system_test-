import { memo } from 'react';

const Loader = memo(({ text = 'Loading...' }) => {
  return (
    <div className="flex flex-col items-center justify-center py-12 sm:py-16 lg:py-20 gap-3 sm:gap-4">
      <div className="w-10 h-10 sm:w-12 sm:h-12 border-4 rounded-full animate-spin"
        style={{ borderColor: 'var(--clr-border)', borderTopColor: 'var(--clr-primary)' }} />
      <p className="text-sm sm:text-base font-medium animate-pulse" style={{ color: 'var(--clr-text-muted)' }}>{text}</p>
    </div>
  );
});

Loader.displayName = 'Loader';

export default Loader;
