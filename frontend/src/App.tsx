import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import AppLayout from './components/Layout/AppLayout';
import ChatView from './views/ChatView';
import DatabasesView from './views/DatabasesView';
import ControlView from './views/ControlView';
import SettingsView from './views/SettingsView';
import LoginView from './views/LoginView';
import UsersView from './views/UsersView';

function ProtectedRoutes() {
  const { user, loading, isAdmin } = useAuth();

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>
        Loading session...
      </div>
    );
  }

  return (
    <Routes>
      <Route path="/login" element={user ? <Navigate to="/databases" replace /> : <LoginView />} />
      <Route path="/accept-invite" element={<LoginView />} />
      <Route path="/" element={user ? <AppLayout /> : <Navigate to="/login" replace />}>
        <Route index element={<Navigate to="/databases" replace />} />
        <Route path="databases" element={<DatabasesView />} />
        <Route path="chat" element={<ChatView />} />
        <Route path="control" element={<ControlView />} />
        <Route path="settings" element={<SettingsView />} />
        <Route path="users" element={isAdmin ? <UsersView /> : <Navigate to="/databases" replace />} />
        <Route path="*" element={<Navigate to="/databases" replace />} />
      </Route>
    </Routes>
  );
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <ProtectedRoutes />
      </Router>
    </AuthProvider>
  );
}

export default App;
