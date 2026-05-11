export function PageShell({ children, className = '', contentClassName = '', backdrop = true }) {
  return (
    <div
      className={`relative min-h-screen overflow-hidden bg-[var(--color-bg-primary)] text-[var(--color-text-primary)] ${className}`}
    >
      {backdrop && (
        <>
          <div
            className="pointer-events-none absolute inset-0 opacity-70"
            style={{
              background:
                'radial-gradient(circle at top left, color-mix(in srgb, var(--color-accent) 10%, transparent) 0, transparent 38%), radial-gradient(circle at top right, color-mix(in srgb, var(--color-accent) 7%, transparent) 0, transparent 32%), linear-gradient(180deg, color-mix(in srgb, var(--color-bg-secondary) 65%, transparent) 0%, transparent 28%)',
            }}
          />
          <div
            className="pointer-events-none absolute inset-x-0 top-0 h-24"
            style={{ background: 'linear-gradient(180deg, var(--color-bg-card), transparent)' }}
          />
        </>
      )}

      <div className={`relative z-10 ${contentClassName}`}>{children}</div>
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  subtitle,
  actions,
  leading,
  containerClassName = 'max-w-6xl',
  className = '',
}) {
  return (
    <header className={`border-b border-[var(--color-border)] bg-[var(--color-bg-card)]/92 backdrop-blur ${className}`}>
      <div className={`mx-auto ${containerClassName} px-4 sm:px-6 py-4 flex flex-wrap items-center justify-between gap-4`}>
        <div className="min-w-0 flex items-center gap-3">
          {leading}
          <div className="min-w-0">
            {eyebrow && (
              <p className="text-xs uppercase tracking-[0.24em] text-[var(--color-text-muted)]">
                {eyebrow}
              </p>
            )}
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="min-w-0 truncate font-heading text-2xl font-bold text-[var(--color-text-primary)]">
                {title}
              </h1>
            </div>
            {subtitle && (
              <p className="mt-1 max-w-2xl text-sm sm:text-base text-[var(--color-text-secondary)]">
                {subtitle}
              </p>
            )}
          </div>
        </div>

        {actions && <div className="flex flex-wrap items-center gap-3">{actions}</div>}
      </div>
    </header>
  );
}