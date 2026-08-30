import { HashRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import AppLayout from './components/Layout/AppLayout';
import ChatView from './views/ChatView';
import DatabasesView from './views/DatabasesView';
import ControlView from './views/ControlView';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<Navigate to="/databases" replace />} />
          <Route path="databases" element={<DatabasesView />} />
          <Route path="chat" element={<ChatView />} />
          <Route path="control" element={<ControlView />} />
          <Route path="*" element={<Navigate to="/databases" replace />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
