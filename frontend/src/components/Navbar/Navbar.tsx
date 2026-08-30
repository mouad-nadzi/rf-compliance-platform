import { useState, useEffect, useRef } from 'react';
import { NavLink } from 'react-router-dom';
import { Bell, CheckCheck, Trash2 } from 'lucide-react';
import { api } from '../../api';
import './Navbar.css';

const Navbar = () => {
  const [unreadCount, setUnreadCount] = useState(0);
  const [notifications, setNotifications] = useState<any[]>([]);
  const [showNotifications, setShowNotifications] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const fetchNotifications = async () => {
    try {
      const res = await api.getNotifications();
      setUnreadCount(res.unread_count || 0);
      setNotifications(res.notifications || []);
    } catch (err) {
      console.error("Error fetching notifications:", err);
    }
  };

  useEffect(() => {
    fetchNotifications();
    const interval = setInterval(fetchNotifications, 10000);
    return () => clearInterval(interval);
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowNotifications(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleToggleNotifications = async () => {
    const nextState = !showNotifications;
    setShowNotifications(nextState);
    if (nextState) {
      // Automatically mark notifications read and clear red badge counter when user clicks the notification bell
      setUnreadCount(0);
      setNotifications(prev => prev.map(n => ({ ...n, is_read: true })));
      try {
        await api.markNotificationsRead();
        await fetchNotifications();
        setUnreadCount(0);
      } catch (err) {
        console.error("Error marking notifications read on open:", err);
      }
    }
  };

  const handleMarkAllRead = async () => {
    try {
      await api.markNotificationsRead();
      setUnreadCount(0);
      setNotifications(prev => prev.map(n => ({ ...n, is_read: true })));
    } catch (err) {
      console.error("Error marking notifications read:", err);
    }
  };

  const handleClearAll = async () => {
    try {
      await api.clearNotifications();
      setUnreadCount(0);
      setNotifications([]);
    } catch (err) {
      console.error("Error clearing notifications:", err);
    }
  };

  return (
    <header className="navbar">
      {/* Left: Stellantis Logo */}
      <div className="navbar-brand">
        <img 
          src="/stellantis.png" 
          alt="Stellantis Logo" 
          className="navbar-logo"
        />
      </div>

      {/* Center: Top Navigation Pills */}
      <nav className="navbar-nav">
        <NavLink 
          to="/databases" 
          className={({ isActive }) => `nav-pill ${isActive ? 'active' : ''}`}
        >
          <span>HOME</span>
        </NavLink>

        <NavLink 
          to="/chat" 
          className={({ isActive }) => `nav-pill ${isActive ? 'active' : ''}`}
        >
          <span>ASSISTANT</span>
        </NavLink>

        <NavLink 
          to="/control" 
          className={({ isActive }) => `nav-pill ${isActive ? 'active' : ''}`}
        >
          <span>AUTOMATIONS</span>
        </NavLink>
      </nav>
      
      {/* Right: Notification Bell & User Admin Badge */}
      <div className="navbar-actions" ref={dropdownRef} style={{ position: 'relative', gap: '0.75rem' }}>
        {/* Circular Notification Bell Button */}
        <button
          onClick={handleToggleNotifications}
          className="notification-bell-btn"
          title="Notifications"
          style={{
            position: 'relative',
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            backgroundColor: showNotifications ? 'rgba(36, 56, 129, 0.12)' : 'var(--bg-body)',
            border: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--brand-blue)',
            cursor: 'pointer',
            transition: 'all 0.2s ease-in-out'
          }}
        >
          <Bell size={20} />
          {unreadCount > 0 && (
            <span
              style={{
                position: 'absolute',
                top: '-2px',
                right: '-2px',
                backgroundColor: '#ef4444',
                color: '#ffffff',
                fontSize: '0.7rem',
                fontWeight: 700,
                borderRadius: '10px',
                padding: '0.1rem 0.35rem',
                minWidth: '18px',
                height: '18px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 2px 5px rgba(239, 68, 68, 0.4)'
              }}
            >
              {unreadCount > 99 ? '99+' : unreadCount}
            </span>
          )}
        </button>

        {/* Notification Popup Dropdown Box */}
        {showNotifications && (
          <div
            style={{
              position: 'absolute',
              top: '50px',
              right: 0,
              width: '360px',
              maxHeight: '440px',
              backgroundColor: 'var(--bg-surface)',
              border: '1px solid var(--border-color)',
              borderRadius: '12px',
              boxShadow: '0 10px 30px rgba(0, 0, 0, 0.15)',
              zIndex: 1000,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden'
            }}
          >
            {/* Header */}
            <div
              style={{
                padding: '0.85rem 1rem',
                borderBottom: '1px solid var(--border-color)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                backgroundColor: 'var(--bg-body)'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '0.9rem', color: 'var(--brand-blue)' }}>
                <Bell size={16} />
                <span>System Notifications</span>
                {unreadCount > 0 && (
                  <span style={{ fontSize: '0.75rem', backgroundColor: '#ef4444', color: '#fff', padding: '0.1rem 0.4rem', borderRadius: '10px' }}>
                    {unreadCount} new
                  </span>
                )}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                {unreadCount > 0 && (
                  <button
                    onClick={handleMarkAllRead}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: 'var(--brand-blue)',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.2rem'
                    }}
                    title="Mark all as read"
                  >
                    <CheckCheck size={14} />
                    <span>Read All</span>
                  </button>
                )}
                {notifications.length > 0 && (
                  <button
                    onClick={handleClearAll}
                    style={{
                      background: 'none',
                      border: 'none',
                      color: '#94a3b8',
                      fontSize: '0.75rem',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center'
                    }}
                    title="Clear all notifications"
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            </div>

            {/* Notification List */}
            <div style={{ overflowY: 'auto', flex: 1, padding: '0.5rem' }}>
              {notifications.length === 0 ? (
                <div style={{ padding: '2rem 1rem', textAlign: 'center', color: '#94a3b8', fontSize: '0.85rem' }}>
                  No notifications recorded yet.
                </div>
              ) : (
                notifications.map((n) => (
                  <div
                    key={n.id}
                    style={{
                      padding: '0.75rem 0.85rem',
                      borderRadius: '8px',
                      marginBottom: '0.4rem',
                      backgroundColor: n.is_read ? 'transparent' : 'rgba(36, 56, 129, 0.04)',
                      borderLeft: n.is_read ? '3px solid transparent' : '3px solid var(--brand-blue)',
                      transition: 'all 0.15s ease'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.25rem' }}>
                      <span style={{ fontWeight: 700, fontSize: '0.82rem', color: 'var(--text-primary)' }}>
                        {n.title}
                      </span>
                      {n.created_at && (
                        <span style={{ fontSize: '0.7rem', color: '#94a3b8', whiteSpace: 'nowrap', marginLeft: '0.5rem' }}>
                          {n.created_at}
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                      {n.message}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* System Admin Badge */}
        <div className="user-profile-badge">
          <span>System Administrator</span>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
