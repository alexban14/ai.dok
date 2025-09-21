import React from "react";
import { Routes, Route } from "react-router-dom";
import ChatPage from "./pages/ChatPage";

const App: React.FC = () => {
    return (
        <div className="min-h-screen bg-gray-100">
            <Routes>
                <Route path="/" element={<ChatPage />} />
            </Routes>
        </div>
    );
};

export default App;