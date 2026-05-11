import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';

function LandingPage() {
  useEffect(() => {
    // Save API URL to local storage so iframe can read it
    localStorage.setItem('VITE_API_URL', import.meta.env.VITE_API_URL || 'https://himasagaru-arxiv-rag-mechanistic-interpretability.hf.space');
  }, []);

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', overflow: 'hidden' }}>
      {/* Floating Header in React */}
      <div style={{
        position: 'absolute',
        top: '20px',
        right: '20px',
        zIndex: 1000,
        display: 'flex',
        gap: '10px'
      }}>
        <Link to="/login" style={{
          padding: '10px 20px',
          background: '#2d6a4f',
          color: 'white',
          borderRadius: '5px',
          textDecoration: 'none',
          fontWeight: 'bold',
          boxShadow: '0 2px 10px rgba(0,0,0,0.2)'
        }}>
          Sign In
        </Link>
        <Link to="/register" style={{
          padding: '10px 20px',
          background: 'white',
          color: '#2d6a4f',
          borderRadius: '5px',
          textDecoration: 'none',
          fontWeight: 'bold',
          border: '1px solid #2d6a4f',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
        }}>
          Sign Up
        </Link>
      </div>

      {/* Iframe loading the old landing page */}
      <iframe
        src="/landing.html"
        title="Landing Page"
        style={{ width: '100%', height: '100%', border: 'none' }}
      />
    </div>
  );
}

export default LandingPage;
