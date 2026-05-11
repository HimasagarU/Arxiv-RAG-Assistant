import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ThemeToggle from '../components/ThemeToggle';
import { PageHeader, PageShell } from '../components/PageShell';

export default function LoginPage() {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(email, password);
    } catch (err) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageShell>
      <PageHeader
        eyebrow="Account"
        title="Welcome back"
        subtitle="Sign in to your research assistant"
        containerClassName="max-w-4xl"
        actions={(
          <>
            <ThemeToggle />
            <Link to="/" className="btn-ghost text-sm">
              ← Home
            </Link>
          </>
        )}
      />

      <div className="flex items-center justify-center px-4 py-10 sm:py-16">
        <div className="glass-card relative z-10 w-full max-w-md animate-fade-in p-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2"
              style={{ fontFamily: 'var(--font-heading)', color: 'var(--color-text-primary)' }}>
            ArXiv RAG
          </h1>
          <p style={{ color: 'var(--color-text-secondary)' }}>
            Sign in to your research assistant
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium mb-1.5"
                   style={{ color: 'var(--color-text-secondary)' }}>
              Email
            </label>
            <input
              id="login-email"
              type="email"
              className="input-field"
              placeholder="researcher@university.edu"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1.5"
                   style={{ color: 'var(--color-text-secondary)' }}>
              Password
            </label>
            <input
              id="login-password"
              type="password"
              className="input-field"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          {error && (
            <div className="p-3 rounded-lg text-sm"
                 style={{ background: 'rgba(248, 113, 113, 0.1)', color: 'var(--color-error)', border: '1px solid rgba(248, 113, 113, 0.2)' }}>
              {error}
            </div>
          )}

          <button
            id="login-submit"
            type="submit"
            className="btn-primary w-full"
            disabled={loading}
          >
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <p className="text-center mt-6 text-sm" style={{ color: 'var(--color-text-muted)' }}>
          Don't have an account?{' '}
          <Link to="/register" className="font-medium" style={{ color: 'var(--color-accent)' }}>
            Create one
          </Link>
        </p>

        <div className="mt-4 text-center">
          <Link to="/how-it-works" className="btn-soft text-sm px-4 py-2" style={{ color: 'var(--color-text-secondary)' }}>
            How it works ↗
          </Link>
        </div>
        </div>
      </div>
    </PageShell>
  );
}
