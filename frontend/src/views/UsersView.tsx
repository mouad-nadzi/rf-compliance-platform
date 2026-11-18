import { useState, useEffect } from 'react';
import { UserPlus, Trash2, Copy, Check, Shield, Users, KeyRound, RefreshCw } from 'lucide-react';
import { api, type ManagedUser } from '../api';
import { useAuth } from '../context/AuthContext';

const UsersView = () => {
  const { user, refreshUser } = useAuth();
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [loading, setLoading] = useState(true);

  // Invite form state
  const [inviteName, setInviteName] = useState('');
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState<'editor' | 'viewer'>('editor');
  const [inviting, setInviting] = useState(false);
  const [inviteResult, setInviteResult] = useState<{ url: string; emailSent: boolean; emailError?: string | null; email: string } | null>(null);
  const [copiedUrl, setCopiedUrl] = useState(false);

  // Admin self-account update state
  const [adminName, setAdminName] = useState(user?.full_name || '');
  const [adminEmail, setAdminEmail] = useState(user?.email || '');
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [savingAccount, setSavingAccount] = useState(false);
  const [statusMsg, setStatusMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const loadUsers = async () => {
    setLoading(true);
    try {
      const res = await api.getUsers();
      setUsers(res.users || []);
    } catch (err: any) {
      setStatusMsg({ type: 'error', text: err?.message || 'Failed to load users.' });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadUsers();
  }, []);

  useEffect(() => {
    if (user) {
      setAdminName(user.full_name);
      setAdminEmail(user.email);
    }
  }, [user]);

  const handleInviteSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setInviting(true);
    setStatusMsg(null);
    setInviteResult(null);
    try {
      const res = await api.inviteUser(inviteEmail, inviteName, inviteRole);
      setInviteResult({
        url: res.invite_url,
        emailSent: res.email_sent,
        emailError: res.email_error,
        email: res.user.email,
      });
      setInviteName('');
      setInviteEmail('');
      await loadUsers();
    } catch (err: any) {
      setStatusMsg({ type: 'error', text: err?.message || 'Failed to create user invitation.' });
    } finally {
      setInviting(false);
    }
  };

  const handleCopyLink = async (url: string) => {
    try {
      await navigator.clipboard.writeText(url);
      setCopiedUrl(true);
      setTimeout(() => setCopiedUrl(false), 2000);
    } catch {
      // ignore clipboard errors
    }
  };

  const handleRoleChange = async (userId: string, role: 'editor' | 'viewer') => {
    setStatusMsg(null);
    try {
      await api.updateUserRole(userId, role);
      await loadUsers();
      setStatusMsg({ type: 'success', text: 'User role updated.' });
    } catch (err: any) {
      setStatusMsg({ type: 'error', text: err?.message || 'Failed to update role.' });
    }
  };

  const handleDeleteUser = async (target: ManagedUser) => {
    if (!window.confirm(`Remove user "${target.full_name}" (${target.email}) and their chat sessions?`)) return;
    setStatusMsg(null);
    try {
      await api.deleteUser(target.id);
      await loadUsers();
      setStatusMsg({ type: 'success', text: `Removed user ${target.email}.` });
    } catch (err: any) {
      setStatusMsg({ type: 'error', text: err?.message || 'Failed to remove user.' });
    }
  };

  const handleUpdateAdminAccount = async (e: React.FormEvent) => {
    e.preventDefault();
    setSavingAccount(true);
    setStatusMsg(null);
    try {
      await api.updateMe({
        full_name: adminName,
        email: adminEmail,
        current_password: currentPassword || undefined,
        new_password: newPassword || undefined,
      });
      setCurrentPassword('');
      setNewPassword('');
      await refreshUser();
      await loadUsers();
      setStatusMsg({ type: 'success', text: 'Administrator account updated.' });
    } catch (err: any) {
      setStatusMsg({ type: 'error', text: err?.message || 'Failed to update Administrator account.' });
    } finally {
      setSavingAccount(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', width: '100%' }}>
      {statusMsg && (
        <div className={`badge ${statusMsg.type === 'success' ? 'badge-green' : 'badge-red'}`} style={{ padding: '0.65rem 1rem', fontSize: '0.82rem' }}>
          {statusMsg.text}
        </div>
      )}

      {/* Invite New User Card */}
      <div className="card" style={{ padding: '1.35rem', border: '1px solid #000000', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <UserPlus size={18} color="var(--brand-blue)" />
          <h2 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: 'var(--brand-blue)' }}>
            Invite New User
          </h2>
        </div>

        <form onSubmit={handleInviteSubmit} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.85rem', alignItems: 'end' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            <label style={{ fontSize: '0.78rem', fontWeight: 600 }}>Full Name</label>
            <input
              type="text"
              className="input"
              placeholder="Jane Doe"
              value={inviteName}
              onChange={(e) => setInviteName(e.target.value)}
              required
              style={{ border: '1px solid #000000' }}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            <label style={{ fontSize: '0.78rem', fontWeight: 600 }}>Email Address</label>
            <input
              type="email"
              className="input"
              placeholder="jane.doe@stellantis.com"
              value={inviteEmail}
              onChange={(e) => setInviteEmail(e.target.value)}
              required
              style={{ border: '1px solid #000000' }}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            <label style={{ fontSize: '0.78rem', fontWeight: 600 }}>Role</label>
            <select
              className="input"
              value={inviteRole}
              onChange={(e) => setInviteRole(e.target.value as 'editor' | 'viewer')}
              style={{ border: '1px solid #000000' }}
            >
              <option value="editor">Allow to Edit (Editor)</option>
              <option value="viewer">Read Only (Viewer)</option>
            </select>
          </div>

          <button type="submit" className="btn btn-primary" disabled={inviting} style={{ justifyContent: 'center', height: '38px', border: '1px solid #000000' }}>
            <UserPlus size={15} />
            {inviting ? 'Sending Invite...' : 'Send Invite Link'}
          </button>
        </form>

        {inviteResult && (
          <div style={{ padding: '0.85rem', borderRadius: '8px', backgroundColor: 'var(--bg-body)', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ fontSize: '0.82rem', fontWeight: 600 }}>
              {inviteResult.emailSent
                ? `Invitation email sent to ${inviteResult.email}.`
                : `Invitation link generated for ${inviteResult.email} (email not sent: ${inviteResult.emailError || 'SMTP not configured'}).`}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <input
                type="text"
                readOnly
                value={inviteResult.url}
                className="input"
                style={{ flex: 1, fontSize: '0.78rem', border: '1px solid #000000' }}
              />
              <button type="button" className="btn btn-secondary" onClick={() => handleCopyLink(inviteResult.url)} style={{ border: '1px solid #000000' }}>
                {copiedUrl ? <Check size={14} /> : <Copy size={14} />}
                {copiedUrl ? 'Copied' : 'Copy Link'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Registered Users Table */}
      <div className="card" style={{ padding: '1.35rem', border: '1px solid #000000', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Users size={18} color="var(--brand-blue)" />
            <h2 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: 'var(--brand-blue)' }}>
              Platform Users ({users.length})
            </h2>
          </div>
          <button className="btn btn-secondary" onClick={loadUsers} style={{ border: '1px solid #000000', fontSize: '0.78rem' }}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>

        {loading ? (
          <div style={{ padding: '1rem', color: 'var(--text-tertiary)' }}>Loading users...</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.84rem' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid var(--border-color)', textAlign: 'left' }}>
                  <th style={{ padding: '0.6rem' }}>Name</th>
                  <th style={{ padding: '0.6rem' }}>Email</th>
                  <th style={{ padding: '0.6rem' }}>Role</th>
                  <th style={{ padding: '0.6rem' }}>Status</th>
                  <th style={{ padding: '0.6rem' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => {
                  const isSelfAdmin = u.role === 'admin';
                  const pendingInviteUrl = u.invite_token ? `${window.location.origin}/#/accept-invite?token=${u.invite_token}` : '';
                  return (
                    <tr key={u.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '0.6rem', fontWeight: 600 }}>{u.full_name}</td>
                      <td style={{ padding: '0.6rem' }}>{u.email}</td>
                      <td style={{ padding: '0.6rem' }}>
                        {isSelfAdmin ? (
                          <span className="badge badge-blue" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                            <Shield size={12} /> Administrator (Unique)
                          </span>
                        ) : (
                          <select
                            className="input"
                            value={u.role}
                            onChange={(e) => handleRoleChange(u.id, e.target.value as 'editor' | 'viewer')}
                            style={{ padding: '0.25rem 0.45rem', fontSize: '0.8rem', border: '1px solid #000000' }}
                          >
                            <option value="editor">Allow to Edit</option>
                            <option value="viewer">Read Only</option>
                          </select>
                        )}
                      </td>
                      <td style={{ padding: '0.6rem' }}>
                        <span className={`badge ${u.status === 'active' ? 'badge-green' : 'badge-yellow'}`}>
                          {u.status.toUpperCase()}
                        </span>
                      </td>
                      <td style={{ padding: '0.6rem', display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                        {pendingInviteUrl && (
                          <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => handleCopyLink(pendingInviteUrl)}
                            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem', border: '1px solid #000000' }}
                            title="Copy invite link"
                          >
                            <Copy size={12} /> Copy Link
                          </button>
                        )}
                        {!isSelfAdmin && (
                          <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => handleDeleteUser(u)}
                            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem', color: '#ef4444', borderColor: '#ef4444' }}
                            title="Remove user"
                          >
                            <Trash2 size={12} /> Remove
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Administrator Account Credentials Card */}
      <div className="card" style={{ padding: '1.35rem', border: '1px solid #000000', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <KeyRound size={18} color="var(--brand-blue)" />
          <h2 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: 'var(--brand-blue)' }}>
            Administrator Account Settings
          </h2>
        </div>

        <form onSubmit={handleUpdateAdminAccount} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.85rem', alignItems: 'end' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            <label style={{ fontSize: '0.78rem', fontWeight: 600 }}>Display Name</label>
            <input
              type="text"
              className="input"
              value={adminName}
              onChange={(e) => setAdminName(e.target.value)}
              required
              style={{ border: '1px solid #000000' }}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            <label style={{ fontSize: '0.78rem', fontWeight: 600 }}>Admin Email</label>
            <input
              type="email"
              className="input"
              value={adminEmail}
              onChange={(e) => setAdminEmail(e.target.value)}
              required
              style={{ border: '1px solid #000000' }}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            <label style={{ fontSize: '0.78rem', fontWeight: 600 }}>Current Password (to change password)</label>
            <input
              type="password"
              className="input"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              placeholder="Current password"
              style={{ border: '1px solid #000000' }}
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
            <label style={{ fontSize: '0.78rem', fontWeight: 600 }}>New Password</label>
            <input
              type="password"
              className="input"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="Optional (min 6 chars)"
              style={{ border: '1px solid #000000' }}
            />
          </div>

          <button type="submit" className="btn btn-primary" disabled={savingAccount} style={{ justifyContent: 'center', height: '38px', border: '1px solid #000000' }}>
            {savingAccount ? 'Saving...' : 'Update Admin Account'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default UsersView;
