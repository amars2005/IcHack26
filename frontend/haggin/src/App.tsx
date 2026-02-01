import { useState, useEffect } from 'react';
import { Pitch } from './components/Pitch';
import { MenuBar } from './components/MenuBar';
import type { Player, HeatmapData, GenerationStatus } from './types';
import type { Preset } from './presets';
import { INITIAL_PLAYERS, INITIAL_BALL_CARRIER, PITCH_WIDTH, PITCH_HEIGHT } from './constants';
import { generateSituation, AIError, fetchPlayerMetrics } from './llm';
import { PlayerInfo } from './components/PlayerInfo';
import BrandingImage from './components/BrandingImage';

function App() {
  const [players, setPlayers] = useState<Player[]>(INITIAL_PLAYERS);
  const [ballCarrier, setBallCarrier] = useState(INITIAL_BALL_CARRIER);
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus>('idle');
  const [aiRefusalMessage, setAiRefusalMessage] = useState<string | null>(null);
  const [selectedPlayerPosition, setSelectedPlayerPosition] = useState<{ x: number; y: number } | null>(null);
  const [selectedMetrics, setSelectedMetrics] = useState<Record<string, number> | null>(null);
  const [xTResult, setXTResult] = useState<any | null>(null);
  const [xTLoading, setXTLoading] = useState(false);
  type Move = { id: string; type: 'pass' | 'shoot' | 'dribble' | 'other'; targetId?: string | null; description: string; score?: number };
  const [moves, setMoves] = useState<Move[]>([]);
  
  const [teamName, setTeamName] = useState("FC Haggin'");
  const [teamColor, setTeamColor] = useState('#7A28AB');
  const [formation, setFormation] = useState('4-4-2');
  const [editingTeamName, setEditingTeamName] = useState(false);
  const teamInputRef = useRef<HTMLInputElement | null>(null);
  const prevTeamNameRef = useRef(teamName);

  const invertHex = (hex: string) => {
    try {
      const cleaned = hex.replace('#', '');
      const num = parseInt(cleaned, 16);
      const inv = (0xffffff ^ num).toString(16).padStart(6, '0');
      return `#${inv}`;
    } catch (e) { return '#3b82f6'; }
  };
  const opponentColor = invertHex(teamColor);

  const attackers = players.filter((p) => p.type === 'attacker');

  const handlePlayerMove = (id: string, x: number, y: number) => {
    const clampedX = Math.max(0, Math.min(PITCH_WIDTH, x));
    const clampedY = Math.max(0, Math.min(PITCH_HEIGHT, y));
    setPlayers((prev) => prev.map((p) => (p.id === id ? { ...p, position: { x: clampedX, y: clampedY } } : p)));
  };

  useEffect(() => {
    const p = players.find((pl) => pl.id === ballCarrier);
    setSelectedPlayerPosition(p ? p.position : null);
    setSelectedMetrics(null);
    if (!ballCarrier) return;
    let mounted = true;
    fetchPlayerMetrics(ballCarrier).then((m) => { if (mounted) setSelectedMetrics(m); });
    return () => { mounted = false; };
  }, [ballCarrier, players]);

  const handleLoadPreset = (preset: Preset) => {
    setPlayers(preset.players);
    setBallCarrier(preset.ballCarrier);
    setGenerationStatus('idle');
  };

  const handleGenerateCustom = async (situation: string) => {
    setGenerationStatus('loading');
    try {
      const result = await generateSituation(situation);
      setPlayers(result.players);
      setBallCarrier(result.ballCarrier);
      setGenerationStatus('success');
      setTimeout(() => setGenerationStatus('idle'), 3000);
    } catch (error) {
      setGenerationStatus('error');
      if (error instanceof AIError && error.code === 'INAPPROPRIATE_PROMPT') setAiRefusalMessage(error.message);
      setTimeout(() => { setGenerationStatus('idle'); setAiRefusalMessage(null); }, 5000);
    }
  };

  const handleCalculateXT = async () => {
    setXTLoading(true);
    setXTResult(null);
    try {
      const payload = getApiPayload();
      console.log('Sending payload:', payload);
      const response = await fetch('http://localhost:5001/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      console.log('xT Result:', result);
      setXTResult(result);
      // Update heatmap data if available
      if (result.heatmap) {
        setHeatmapData(result.heatmap);
      }
    } catch (error) {
      console.error('xT calculation error:', error);
      setXTResult({ error: 'Failed to calculate xT' });
    } finally {
      setXTLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', width: '100vw', height: '100vh', background: '#020617', color: '#f8fafc', overflow: 'hidden' }}>
      {isLoading && (
        <div style={{ position: 'fixed', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#020617', zIndex: 10000, flexDirection: 'column' }}>
          <div style={{ fontSize: 24, fontWeight: 'bold', color: '#38bdf8' }}>Initializing Tactical View...</div>
        </div>
      )}

      {/* LEFT: PITCH AREA */}
      <main 
        ref={pitchContainerRef}
        style={{ 
          flex: 1, 
          height: '100%', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center', 
          position: 'relative',
          padding: isPitchFullscreen ? 0 : '20px',
          background: isPitchFullscreen ? '#000' : 'transparent',
        }}
      >
        {!isPitchFullscreen && (
          <div style={{ position: 'absolute', left: 64, top: 20, zIndex: 10 }}>
            {editingTeamName ? (
              <input
                ref={teamInputRef}
                value={teamName}
                onChange={(e) => setTeamName(e.target.value)}
                onBlur={() => setEditingTeamName(false)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    setEditingTeamName(false);
                  } else if (e.key === 'Escape') {
                    setTeamName(prevTeamNameRef.current);
                    setEditingTeamName(false);
                  }
                }}
                style={{ margin: 0, fontSize: 36, fontWeight: 900, background: 'transparent', border: '1px solid rgba(255,255,255,0.08)', padding: '6px 10px', borderRadius: 6, color: teamColor }}
              />
            ) : (
              <h1 onClick={() => setEditingTeamName(true)} style={{ margin: 0, fontSize: 36, fontWeight: 900, color: teamColor, cursor: 'pointer' }}>{teamName.toUpperCase()}</h1>
            )}
          </div>
        )}

        <div style={{ position: 'absolute', right: 20, top: 20, zIndex: 50 }}>
          <button
            onClick={async () => {
              if (!document.fullscreenElement) await pitchContainerRef.current?.requestFullscreen();
              else await document.exitFullscreen();
            }}
            style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: '#fff', padding: '10px 15px', borderRadius: 8, cursor: 'pointer' }}
          >
            {isPitchFullscreen ? 'EXIT' : 'FULLSCREEN'}
          </button>
        </div>

        <Pitch
          players={players}
          ballCarrier={ballCarrier}
          onPlayerMove={handlePlayerMove}
          scale={isPitchFullscreen ? Math.min(viewport.w / PITCH_WIDTH, viewport.h / PITCH_HEIGHT) : scale}
          teamColor={teamColor}
          opponentColor={opponentColor}
          onAssignBall={handleAssignBall}
        />
      </main>

      {/* RIGHT: INTEGRATED SCROLLABLE SIDEBAR */}
      {!isPitchFullscreen && (
        <aside style={{ 
          width: sidebarWidth, 
          height: '100%', 
          background: '#0f172a', 
          borderLeft: '1px solid rgba(255,255,255,0.1)', 
          display: 'flex', 
          flexDirection: 'column',
          padding: '24px',
          gap: '20px',
          overflowY: 'auto', 
          zIndex: 100
        }}>
          {/* (duplicate Team Settings removed) */}

          {/* BRANDING: logo / upload */}
          <div style={{ flexShrink: 0 }}>
            <BrandingImage />
          </div>

          {/* BOX 1: SUGGESTED MOVES */}
          <div style={{ background: 'linear-gradient(90deg, rgba(124,58,237,0.08), rgba(99,102,241,0.03))', padding: '16px', borderRadius: '14px', border: '1px solid rgba(124,58,237,0.18)', boxShadow: '0 8px 30px rgba(2,6,23,0.6)', flexShrink: 0 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <h3 style={{ margin: 0, fontSize: 13, color: '#e6eef8', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Top 5 Moves</h3>
              <button onClick={fetchSuggestedMoves} style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.08)', color: '#fff', padding: '6px 8px', borderRadius: 8, cursor: 'pointer', fontSize: 12 }}>Suggest</button>
            </div>
            <ol style={{ margin: 0, paddingLeft: 16, color: '#e6eef8', fontSize: 14 }}>
              {moves.length === 0 ? (
                <li style={{ color: '#cbd5e1' }}>No suggestions yet — click Suggest</li>
              ) : (
                moves.map((m, i) => (
                  <li key={m.id} style={{ marginBottom: 8, display: 'flex', gap: 10, alignItems: 'center', padding: '10px', borderRadius: 10, background: i === 0 ? 'linear-gradient(90deg, rgba(246,201,77,0.18), rgba(255,242,179,0.06))' : 'transparent', border: i === 0 ? '1px solid rgba(246,201,77,0.25)' : 'none', boxShadow: i === 0 ? '0 8px 22px rgba(246,201,77,0.08)' : 'none' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {i === 0 && <span style={{ color: '#f6c94d', fontSize: 18, lineHeight: 1 }}>★</span>}
                      <div style={{ fontWeight: 900, width: 24, color: i === 0 ? '#ffffff' : '#e6eef8' }}>{i + 1}</div>
                    </div>
                    <div style={{ flex: 1, fontSize: 15, fontWeight: i === 0 ? 800 : 600, color: i === 0 ? '#ffffff' : '#e6eef8' }}>{m.description}</div>
                    <div style={{ marginLeft: 8, display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ fontSize: 11, padding: '5px 10px', borderRadius: 999, background: i === 0 ? '#fff7e6' : '#0b1224', color: i === 0 ? '#704c00' : '#c7b3ff', border: i === 0 ? '1px solid rgba(124,58,237,0.06)' : '1px solid rgba(124,58,237,0.2)' }}>{m.type}</span>
                      {m.score != null && <span style={{ fontSize: 12, color: i === 0 ? '#6b5a00' : '#9ca3af' }}>{m.score.toFixed(2)}</span>}
                    </div>
                  </li>
                ))
              )}
            </ol>
          </div>

          {/* BOX 2: PLAYER INFO (moved to top) */}
          <div style={{ flexShrink: 0 }}>
            <PlayerInfo
              playerId={ballCarrier}
              playerPosition={selectedPlayerPosition || undefined}
              metrics={selectedMetrics}
              onClose={undefined}
            />
          </div>

          {/* BOX 2: TEAM SETTINGS */}
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '20px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', flexShrink: 0 }}>
            <h2 style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', marginBottom: '16px', letterSpacing: '0.05em' }}>Team Settings</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <input value={teamName} onChange={(e) => setTeamName(e.target.value)} placeholder="Team Name" style={{ width: '100%', background: '#1e293b', border: '1px solid #334155', color: '#fff', padding: '10px', borderRadius: '6px' }} />
              <div style={{ display: 'flex', gap: '10px' }}>
                <input type="color" value={teamColor} onChange={(e) => setTeamColor(e.target.value)} style={{ width: '45px', height: '40px', border: 'none', background: 'none', cursor: 'pointer' }} />
                <select value={formation} onChange={(e) => {
                  const val = e.target.value;
                  setFormation(val);
                  const preset = FORMATION_PRESETS.find(p => p.name === val);
                  if (preset) handleLoadPreset(preset);
                }} style={{ flex: 1, background: '#1e293b', border: '1px solid #334155', color: '#fff', borderRadius: '6px', padding: '0 10px' }}>
                  {FORMATION_PRESETS.map(p => <option key={p.name} value={p.name}>{p.name}</option>)}
                </select>
              </div>
            </div>
          </div>

          {/* BOX 2: TACTICAL MENU (inlined) */}
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '20px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', flexShrink: 0 }}>
            <h2 style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', marginBottom: '12px', letterSpacing: '0.05em' }}>Controls</h2>

            {/* (Ball possession and Team Colors removed) */}

            {/* xT calculation */}
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: '#9ca3af' }}>Expected Threat (xT)</label>
              <p style={{ fontSize: '12px', color: '#9ca3af', marginTop: 0 }}>Calculate xT for the player currently in possession.</p>
              <button
                onClick={handleCalculateXT}
                disabled={xTLoading}
                style={{ marginTop: '8px', padding: '10px', background: xTLoading ? '#374151' : '#f59e0b', color: 'white', border: 'none', borderRadius: '6px', cursor: xTLoading ? 'not-allowed' : 'pointer', fontSize: '13px', width: '100%' }}
              >
                {xTLoading ? 'Calculating…' : 'Calculate xT for ball carrier'}
              </button>

              {xTResult && (
                <div style={{ marginTop: 10, padding: 8, background: '#071025', borderRadius: 6, fontSize: 12 }}>
                  <div style={{ fontWeight: 700, marginBottom: 6 }}>xT result</div>
                  <pre style={{ whiteSpace: 'pre-wrap', margin: 0, fontSize: 11 }}>{JSON.stringify(xTResult, null, 2)}</pre>
                </div>
              )}
            </div>

            {/* Custom Situation Generator */}
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '13px', marginBottom: '6px', color: '#9ca3af' }}>Custom Situation</label>
              <textarea
                value={customSituation}
                onChange={(e) => setCustomSituation(e.target.value)}
                placeholder="Describe an attacking situation (e.g., 'Quick throw-in near the penalty box with numbers up')"
                style={{ width: '100%', padding: '8px', background: '#0b1224', color: 'white', border: '1px solid #273449', borderRadius: '6px', fontSize: '13px', minHeight: '72px', resize: 'vertical', fontFamily: 'inherit' }}
              />
              <button
                onClick={async () => { if (!customSituation.trim()) return; await handleGenerateCustom(customSituation); }}
                disabled={!customSituation.trim() || generationStatus === 'loading'}
                style={{ marginTop: '8px', padding: '10px', background: generationStatus === 'loading' ? '#374151' : '#7c3aed', color: 'white', border: 'none', borderRadius: '6px', cursor: customSituation.trim() && generationStatus !== 'loading' ? 'pointer' : 'not-allowed', fontSize: '13px', width: '100%', opacity: customSituation.trim() && generationStatus !== 'loading' ? 1 : 0.7 }}
              >
                {generationStatus === 'loading' ? 'Generating...' : 'Generate with AI'}
              </button>

              {generationStatus === 'success' && (
                <div style={{ marginTop: '8px', padding: '8px', background: '#065f46', borderRadius: '6px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '16px' }}>✓</span>
                  <span>Successfully generated positions!</span>
                </div>
              )}

              {generationStatus === 'error' && !aiRefusalMessage && (
                <div style={{ marginTop: '8px', padding: '8px', background: '#7f1d1d', borderRadius: '6px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '16px' }}>✗</span>
                  <span>Generation failed. Check backend connection.</span>
                </div>
              )}

              {aiRefusalMessage && (
                <div style={{ marginTop: '8px', padding: '8px', background: '#92400e', borderRadius: '6px', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '16px' }}>⚠</span>
                  <div>
                    <div style={{ fontWeight: '600' }}>AI refused to answer</div>
                    <div style={{ fontSize: '12px', color: '#fde68a' }}>{aiRefusalMessage}</div>
                  </div>
                </div>
              )}
            </div>

            {/* Attacking Situations (simple presets) */}
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', fontSize: '13px', marginBottom: '8px', color: '#9ca3af' }}>Attacking Situations</label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {SITUATION_PRESETS.map((preset) => (
                  <button key={preset.name} onClick={() => handleLoadPreset(preset)} style={{ padding: '8px', background: '#065f46', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '13px' }}>{preset.name}</button>
                ))}
              </div>
            </div>
          </div>

          
        </aside>
      )}
    </div>
  );
}

export default App;
