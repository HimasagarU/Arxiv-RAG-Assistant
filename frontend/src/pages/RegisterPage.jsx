import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ThemeToggle from '../components/ThemeToggle';
import { PageHeader, PageShell } from '../components/PageShell';

export default function RegisterPage() {
  const { register } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }
    setLoading(true);
    try {
      await register(email, password, displayName || 'Researcher');
    } catch (err) {
      setError(err.message || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageShell>
      <PageHeader
        eyebrow="Account"
        title="Create your account"
        subtitle="Set up a research workspace for ArXiv RAG Assistant"
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
            Join ArXiv RAG
          </h1>
          <p style={{ color: 'var(--color-text-secondary)' }}>
            Create your research assistant account
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium mb-1.5"
                   style={{ color: 'var(--color-text-secondary)' }}>
              Display Name
            </label>
            <input
              id="register-name"
              type="text"
              className="input-field"
              placeholder="Dr. Jane Smith"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1.5"
                   style={{ color: 'var(--color-text-secondary)' }}>
              Email
            </label>
            <input
              id="register-email"
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
              id="register-password"
              type="password"
              className="input-field"
              placeholder="Min 8 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
            />
          </div>

          {error && (
            <div className="p-3 rounded-lg text-sm"
                 style={{ background: 'rgba(248, 113, 113, 0.1)', color: 'var(--color-error)', border: '1px solid rgba(248, 113, 113, 0.2)' }}>
              {error}
            </div>
          )}

          <button
            id="register-submit"
            type="submit"
            className="btn-primary w-full"
            disabled={loading}
          >
            {loading ? 'Creating account...' : 'Create Account'}
          </button>
        </form>

        <p className="text-center mt-6 text-sm" style={{ color: 'var(--color-text-muted)' }}>
          Already have an account?{' '}
          <Link to="/login" className="font-medium" style={{ color: 'var(--color-accent)' }}>
            Sign in
          </Link>
        </p>
        </div>
      </div>
    </PageShell>
  );
}
