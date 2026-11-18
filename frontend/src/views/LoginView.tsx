import { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Lock, Mail, UserCheck, Loader2, Shield } from 'lucide-react';
import { api } from '../api';
import { useAuth } from '../context/AuthContext';

const LoginView = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { login, setAuthUser } = useAuth();

  const searchParams = new URLSearchParams(location.search);
  const inviteToken = location.pathname.includes('accept-invite') ? (searchParams.get('token') || '') : '';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [inviteInfo, setInviteInfo] = useState<{ email: string; full_name: string; role: string } | null>(null);
  const [inviteError, setInviteError] = useState<string | null>(null);
  const [validatingInvite, setValidatingInvite] = useState(Boolean(inviteToken));

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!inviteToken) return;
    setValidatingInvite(true);
    setInviteError(null);
    api.validateInvite(inviteToken)
      .then((info) => {
        setInviteInfo(info);
        setEmail(info.email);
        setFullName(info.full_name || '');
      })
      .catch((err: any) => {
        setInviteError(err?.message || 'Invalid or expired invitation link.');
      })
      .finally(() => setValidatingInvite(false));
  }, [inviteToken]);

  const handleLoginSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(email, password);
      navigate('/databases', { replace: true });
    } catch (err: any) {
      setError(err?.message || 'Sign in failed.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleAcceptInviteSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (password.length < 6) {
      setError('Password must be at least 6 characters.');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    setSubmitting(true);
    try {
      const res = await api.acceptInvite(inviteToken, password, fullName);
      setAuthUser(res.user);
      navigate('/databases', { replace: true });
    } catch (err: any) {
      setError(err?.message || 'Could not activate account.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      width: '100%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: 'var(--bg-body)',
      padding: '1.5rem'
    }}>
      <div className="card" style={{
        width: '100%',
        maxWidth: '420px',
        padding: '2rem',
        border: '1px solid #000000',
        display: 'flex',
        flexDirection: 'column',
        gap: '1.25rem'
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.6rem', textAlign: 'center' }}>
          <img src="/stellantis.png" alt="Stellantis" style={{ height: '34px', objectFit: 'contain' }} />
          <h2 style={{ margin: 0, fontSize: '1.2rem', color: 'var(--brand-blue)', fontWeight: 700 }}>
            RF Compliance Platform
          </h2>
          <p className="text-secondary" style={{ margin: 0, fontSize: '0.82rem' }}>
            {inviteToken ? 'Complete your account invitation' : 'Sign in to access compliance intelligence'}
          </p>
        </div>

        {inviteToken ? (
          validatingInvite ? (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', padding: '1.5rem', color: 'var(--text-secondary)' }}>
              <Loader2 size={18} className="spin-icon" /> Verifying invitation...
            </div>
          ) : inviteError ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              <div className="badge badge-red" style={{ padding: '0.75rem', fontSize: '0.82rem', textAlign: 'center' }}>
                {inviteError}
              </div>
              <button className="btn btn-secondary" onClick={() => navigate('/login')}>
                Back to Sign In
              </button>
            </div>
          ) : (
            <form onSubmit={handleAcceptInviteSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
              <div style={{ padding: '0.75rem', borderRadius: '8px', backgroundColor: 'var(--bg-body)', border: '1px solid var(--border-color)', fontSize: '0.82rem' }}>
                <div><strong>Email:</strong> {inviteInfo?.email}</div>
                <div style={{ marginTop: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <strong>Assigned Role:</strong>
                  <span className="badge badge-blue" style={{ fontSize: '0.72rem' }}>
                    {inviteInfo?.role === 'editor' ? 'Allow to Edit' : 'Read Only'}
                  </span>
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                <label style={{ fontSize: '0.78rem', fontWeight: 600 }}>Full Name</label>
                <input
                  type="text"
                  className="input"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  required
                  style={{ border: '1px solid #000000' }}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                <label style={{ fontSize: '0.78rem', fontWeight: 600 }}>Set Password (min 6 chars)</label>
                <input
                  type="password"
                  className="input"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={6}
                  style={{ border: '1px solid #000000' }}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                <label style={{ fontSize: '0.78rem', fontWeight: 600 }}>Confirm Password</label>
                <input
                  type="password"
                  className="input"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                  minLength={6}
                  style={{ border: '1px solid #000000' }}
                />
              </div>

              {error && (
                <div className="badge badge-red" style={{ padding: '0.55rem', fontSize: '0.78rem', textAlign: 'center' }}>
                  {error}
                </div>
              )}

              <button type="submit" className="btn btn-primary" disabled={submitting} style={{ justifyContent: 'center', padding: '0.65rem', border: '1px solid #000000' }}>
                <UserCheck size={16} />
                {submitting ? 'Activating Account...' : 'Activate & Sign In'}
              </button>
            </form>
          )
        ) : (
          <form onSubmit={handleLoginSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.95rem' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
              <label style={{ fontSize: '0.78rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Mail size={14} /> Email Address
              </label>
              <input
                type="email"
                className="input"
                placeholder="name@stellantis.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                style={{ border: '1px solid #000000' }}
              />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
              <label style={{ fontSize: '0.78rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Lock size={14} /> Password
              </label>
              <input
                type="password"
                className="input"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                style={{ border: '1px solid #000000' }}
              />
            </div>

            {error && (
              <div className="badge badge-red" style={{ padding: '0.55rem', fontSize: '0.78rem', textAlign: 'center' }}>
                {error}
              </div>
            )}

            <button type="submit" className="btn btn-primary" disabled={submitting} style={{ justifyContent: 'center', padding: '0.65rem', border: '1px solid #000000' }}>
              <Shield size={16} />
              {submitting ? 'Signing In...' : 'Sign In'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
};

export default LoginView;
