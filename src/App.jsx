import React, { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import ChatBot from "./components/ChatBot";

import Dashboard from "./pages/Dashboard";
import Upload from "./pages/Upload";
import Predictions from "./pages/Predictions";
import Analytics from "./pages/Analytics";
import History from "./pages/History";
import Login from "./pages/Login";

function App() {
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    // If not authenticated, render only the Login page, intercepting all routes
    if (!isAuthenticated) {
        return (
            <BrowserRouter>
                <Routes>
                    <Route path="*" element={<Login onLogin={() => setIsAuthenticated(true)} />} />
                </Routes>
            </BrowserRouter>
        );
    }

    // Authenticated layout
    return (
        <BrowserRouter>
            <div className="app-layout">
                <Sidebar />
                <div className="main-content">
                    <Navbar onLogout={() => setIsAuthenticated(false)} />
                    <div className="page-content" style={{ backgroundColor: 'var(--bg-color)' }}>
                        <Routes>
                            <Route path="/" element={<Dashboard />} />
                            <Route path="/upload" element={<Upload />} />
                            <Route path="/predictions" element={<Predictions />} />
                            <Route path="/analytics" element={<Analytics />} />
                            <Route path="/history" element={<History />} />
                            {/* Redirect unknown logged-in routes to dashboard */}
                            <Route path="*" element={<Navigate to="/" replace />} />
                        </Routes>
                    </div>
                </div>
                <ChatBot />
            </div>
        </BrowserRouter>
    );

}

export default App;