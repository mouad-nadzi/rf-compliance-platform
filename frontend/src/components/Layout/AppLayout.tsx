import { useState } from 'react';
import { Outlet, useOutletContext } from 'react-router-dom';
import Navbar from '../Navbar/Navbar';
import Sidebar from '../Sidebar/Sidebar';

export type LayoutContextType = {
  selectedTable: string;
  setSelectedTable: (tbl: string) => void;
  customTables: string[];
  setCustomTables: React.Dispatch<React.SetStateAction<string[]>>;
  activeSessionId: string | null;
  setActiveSessionId: (id: string | null) => void;
};

export function useLayoutContext() {
  return useOutletContext<LayoutContextType>();
}

const AppLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedTable, setSelectedTable] = useState('RF Certificates');
  const [customTables, setCustomTables] = useState<string[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  return (
    <div className="app-container" style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <Navbar />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <Sidebar 
          isOpen={sidebarOpen} 
          onToggle={() => setSidebarOpen(prev => !prev)} 
          selectedTable={selectedTable}
          setSelectedTable={setSelectedTable}
          customTables={customTables}
          setCustomTables={setCustomTables}
          activeSessionId={activeSessionId}
          setActiveSessionId={setActiveSessionId}
        />
        <main className="page-content" style={{ flex: 1, padding: '1.5rem', overflowY: 'auto' }}>
          <Outlet context={{ selectedTable, setSelectedTable, customTables, setCustomTables, activeSessionId, setActiveSessionId }} />
        </main>
      </div>
    </div>
  );
};

export default AppLayout;
