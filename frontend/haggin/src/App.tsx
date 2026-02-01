import { useState, useEffect, useRef } from 'react';
import { Pitch } from './components/Pitch';
import type { Player } from './types';
import type { Preset } from './presets';
import { FORMATION_PRESETS, SITUATION_PRESETS } from './presets';
import type { GenerationStatus } from './types';
import { INITIAL_PLAYERS, INITIAL_BALL_CARRIER, PITCH_WIDTH, PITCH_HEIGHT } from './constants';
import { generateSituation, AIError, fetchPlayerMetrics } from './llm';
import { PlayerInfo } from './components/PlayerInfo';
import BrandingImage from './components/BrandingImage.tsx';

function App() {
  const [players, setPlayers] = useState<Player[]>(INITIAL_PLAYERS);
  const [ballCarrier, setBallCarrier] = useState(INITIAL_BALL_CARRIER);
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus>('idle');
  const [aiRefusalMessage, setAiRefusalMessage] = useState<string | null>(null);
  const [customSituation, setCustomSituation] = useState('');
  const [selectedPlayerPosition, setSelectedPlayerPosition] = useState<{ x: number; y: number } | null>(null);
  const [selectedMetrics, setSelectedMetrics] = useState<Record<string, number> | null>(null);
  const [isPitchFullscreen, setIsPitchFullscreen] = useState(false);
  const pitchContainerRef = useRef<HTMLDivElement | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [xTResult, setXTResult] = useState<any | null>(null);
  const [xTLoading, setXTLoading] = useState(false);
  const [testResult, setTestResult] = useState<any | null>(null);
  const [testLoading, setTestLoading] = useState(false);
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

  const handleAssignBall = (id: string) => {
    setBallCarrier(id);
  };

  const fetchSuggestedMoves = async () => {
    // Placeholder: generate simple moves based on current players and ball carrier.
    // Backend will replace this with a real API returning structured Move[].
    const attackersList = players.filter(p => p.type === 'attacker' && p.id !== ballCarrier);
    const sample: Move[] = [];
    // Suggest up to 4 passes to nearest attackers + 1 shoot/dribble option
    for (let i = 0; i < Math.min(4, attackersList.length); i++) {
      const p = attackersList[i];
      sample.push({ id: `m-pass-${p.id}`, type: 'pass', targetId: p.id, description: `Pass to #${p.id}`, score: 0.5 - i * 0.05 });
    }
    sample.push({ id: 'm-dribble', type: 'dribble', description: 'Dribble forward', score: 0.3 });
    setMoves(sample.slice(0,5));
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
    try {
      const attackers = players.filter((p) => p.type === 'attacker').map((p) => ({ x: p.position.x, y: p.position.y, id: p.id, team: 1 }));
      const defenders = players.filter((p) => p.type === 'defender').map((p) => ({ x: p.position.x, y: p.position.y, id: p.id, team: 0 }));
      const keepers = players.filter(p => p.id === '1' || p.id === 'd1').map(p => ({ x: p.position.x, y: p.position.y, id: p.id, team: p.id === '1' ? 1 : 0 }));
      const resp = await fetch('http://localhost:5001/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ attackers, defenders, keepers, ball_id: ballCarrier }) });
      setXTResult(await resp.json());
    } catch (err) { setXTResult({ error: 'xT failed' }); }
    finally { setXTLoading(false); }
  };

  const handleTestEndpoint = async () => {
    setTestLoading(true);
    try {
      const attackers = players.filter((p) => p.type === 'attacker').map((p) => ({ x: p.position.x, y: p.position.y, id: p.id, team: 1 }));
      const defenders = players.filter((p) => p.type === 'defender').map((p) => ({ x: p.position.x, y: p.position.y, id: p.id, team: 0 }));
      const keepers = players.filter(p => p.id === '1' || p.id === 'd1').map(p => ({ x: p.position.x, y: p.position.y, id: p.id, team: p.id === '1' ? 1 : 0 }));
      const resp = await fetch('http://localhost:5001/test', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ attackers, defenders, keepers, ball_id: ballCarrier }) });
      const result = await resp.json();
      setTestResult(result);
      console.log('Test endpoint result:', result);
    } catch (err) { 
      setTestResult({ error: 'Test endpoint failed' }); 
      console.error('Test endpoint error:', err);
    }
    finally { setTestLoading(false); }
  };

  const [viewport, setViewport] = useState({ w: window.innerWidth, h: window.innerHeight });
  useEffect(() => {
    const onResize = () => setViewport({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  // Inject Poppins font for softer, rounded team title
  useEffect(() => {
    if (typeof document === 'undefined') return;
    if (!document.getElementById('poppins-font')) {
      const l = document.createElement('link');
      l.id = 'poppins-font';
      l.rel = 'stylesheet';
      l.href = 'https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,400;0,600;0,700;0,800;1,700;1,800&display=swap';
      document.head.appendChild(l);
    }
  }, []);

  const sidebarWidth = 400; // Slightly wider to accommodate internal menu padding
  const horizontalPadding = 80;
  const verticalPadding = 60;
  const scale = Math.min(
    (viewport.w - sidebarWidth - horizontalPadding) / PITCH_WIDTH,
    (viewport.h - verticalPadding) / PITCH_HEIGHT
  );

  useEffect(() => {
    const onFs = () => setIsPitchFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFs);
    return () => document.removeEventListener('fullscreenchange', onFs);
  }, []);

  useEffect(() => {
    if (editingTeamName) {
      prevTeamNameRef.current = teamName;
      // focus next tick
      setTimeout(() => {
        teamInputRef.current?.focus();
        teamInputRef.current?.select();
      }, 0);
    }
  }, [editingTeamName]);

  useEffect(() => {
    const t = setTimeout(() => setIsLoading(false), 2000);
    return () => clearTimeout(t);
  }, []);

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
          <div style={{ position: 'absolute', left: 64, top: 12, zIndex: 10 }}>
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
                style={{ margin: 0, fontSize: 36, fontWeight: 800, fontFamily: "'Poppins', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial", letterSpacing: '0.02em', background: 'transparent', border: '1px solid rgba(255,255,255,0.08)', padding: '6px 10px', borderRadius: 6, color: teamColor }}
              />
            ) : (
              <h1 onClick={() => setEditingTeamName(true)} style={{ margin: 0, fontSize: 46, fontWeight: 800, fontStyle: 'italic', fontFamily: "'Poppins', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial", letterSpacing: '0.01em', color: teamColor, cursor: 'pointer' }}>{teamName.toUpperCase()}</h1>
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

          {/* BOX 2: PLAYER INFO (moved to top) */}
          <div style={{ flexShrink: 0, background: 'linear-gradient(180deg, rgba(15,23,42,0.7), rgba(11,18,36,0.6))', padding: '12px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.04)', boxShadow: '0 8px 20px rgba(2,6,23,0.6)' }}>
            <PlayerInfo
              playerId={ballCarrier}
              playerPosition={selectedPlayerPosition || undefined}
            />
          </div>

          {/* BOX 2: DECISION ENGINE */}
          <div style={{ background: 'linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%)', padding: '20px', borderRadius: '12px', border: '1px solid #312e81', flexShrink: 0 }}>
            <h2 style={{ fontSize: '16px', color: '#e2e8f0', marginBottom: '8px' }}>
              Decision Engine
            </h2>
            <p style={{ fontSize: '11px', color: '#6b7280', marginTop: 0, marginBottom: '16px' }}>
              AI-powered analysis of optimal actions
            </p>

            {/* Main Analysis Button */}
            <button
              onClick={handleCalculateXT}
              disabled={xTLoading}
              style={{
                padding: '14px 16px',
                background: xTLoading 
                  ? 'linear-gradient(135deg, #4b5563 0%, #374151 100%)' 
                  : 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                cursor: xTLoading ? 'not-allowed' : 'pointer',
                fontSize: '14px',
                fontWeight: 600,
                width: '100%',
                opacity: xTLoading ? 0.7 : 1,
                boxShadow: xTLoading ? 'none' : '0 4px 14px rgba(139, 92, 246, 0.4)',
                transition: 'all 0.2s ease',
              }}
            >
              {xTLoading ? 'Computing...' : 'Compute Best Action'}
            </button>

            {/* Results Display: show ranked best options first, raw xT collapsible */}
            {xTResult && !xTResult.error && (
              <div style={{ marginTop: '16px', background: 'rgba(0,0,0,0.18)', borderRadius: '10px', overflow: 'hidden', border: '1px solid rgba(99, 102, 241, 0.08)' }}>
                {(() => {
                  const passActions = Object.entries(xTResult)
                    .filter(([key]) => key.startsWith('pass_to_'))
                    .map(([key, value]: [string, any]) => ({
                      playerId: key.replace('pass_to_', ''),
                      probability: value['P(success)'],
                      reward: value.reward,
                      risk: value.risk,
                      score: value.score,
                    }))
                    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

                  const options: Array<any> = [];
                  if (xTResult.shoot) {
                    options.push({ id: 'shoot', label: 'Shoot', desc: 'Expected Goal', value: xTResult.shoot.xG, kind: 'shoot' });
                  }

                  passActions.forEach((p) => {
                    options.push({ id: `pass_${p.playerId}`, label: `Pass to #${p.playerId}`, desc: `${(p.probability * 100).toFixed(0)}% success`, value: p.score ?? 0, kind: 'pass', meta: p });
                  });

                  const dribbleMove = moves.find(m => m.type === 'dribble');
                  if (xTResult.dribble) options.push({ id: 'dribble', label: 'Dribble', desc: 'Dribble option', value: xTResult.dribble.score ?? 0, kind: 'dribble' });
                  else if (dribbleMove) options.push({ id: 'dribble', label: 'Dribble', desc: dribbleMove.description, value: dribbleMove.score ?? 0, kind: 'dribble' });

                  options.sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
                  const topOptions = options.slice(0, 4);

                  return (
                    <div style={{ padding: '12px' }}>
                      <div style={{ fontWeight: 700, color: '#c084fc', marginBottom: '8px', fontSize: '13px' }}>Top Recommendations</div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {topOptions.length === 0 && (
                          <div style={{ fontSize: 12, color: '#9ca3af' }}>No suggestions available.</div>
                        )}

                        {topOptions.map((opt, idx) => {
                          const isTop = idx === 0;
                          return (
                            <div
                              key={opt.id}
                              style={{
                                  display: 'flex',
                                  justifyContent: 'space-between',
                                  alignItems: 'center',
                                  padding: isTop ? '14px' : '10px',
                                  borderRadius: '10px',
                                  background: isTop ? 'linear-gradient(135deg, rgba(245,158,11,0.12), rgba(234,179,8,0.06))' : 'rgba(0,0,0,0.16)',
                                  boxShadow: isTop ? '0 6px 18px rgba(245,158,11,0.08)' : 'none',
                                  border: isTop ? '1px solid rgba(245,158,11,0.18)' : '1px solid rgba(255,255,255,0.02)'
                                }}
                            >
                              <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                                <div style={{ width: 30, height: 30, borderRadius: 8, background: isTop ? '#f59e0b' : 'rgba(255,255,255,0.04)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#cbd5e1', fontWeight: 700 }}>{idx + 1}</div>
                                <div>
                                  <div style={{ fontWeight: 800, fontSize: isTop ? 15 : 13 }}>{opt.label}</div>
                                  <div style={{ fontSize: 11, color: '#9ca3af' }}>{opt.desc}</div>
                                </div>
                              </div>
                              <div style={{ fontWeight: 800, color: opt.value > 0 ? '#16a34a' : opt.value < 0 ? '#ef4444' : '#f59e0b', fontSize: isTop ? 16 : 13 }}>
                                {opt.kind === 'shoot' ? `${(opt.value * 100).toFixed(1)}%` : `${opt.value > 0 ? '+' : ''}${(opt.value * 100).toFixed(2)}%`}
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      {/* Full xT results hidden by default to keep UI focused on recommendations */}
                    </div>
                  );
                })()}
              </div>
            )}

            {xTResult?.error && (
              <div style={{ marginTop: '12px', padding: '10px', background: '#7f1d1d', borderRadius: '6px', fontSize: '12px' }}>
                <span>Error: {xTResult.error}</span>
              </div>
            )}
          </div>

          {/* BOX 3: AI SITUATION GENERATOR */}
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <h2 style={{ fontSize: '13px', color: '#9ca3af', marginBottom: '10px' }}>
              AI Scenario Generator
            </h2>
            <textarea
              value={customSituation}
              onChange={(e) => setCustomSituation(e.target.value)}
              placeholder="Describe an attacking situation (e.g., 'Quick throw-in near the penalty box with numbers up')"
              style={{ width: '100%', padding: '8px', background: '#0b1224', color: 'white', border: '1px solid #273449', borderRadius: '6px', fontSize: '12px', minHeight: '60px', resize: 'vertical', fontFamily: 'inherit' }}
            />
            <button
              onClick={async () => { if (!customSituation.trim()) return; await handleGenerateCustom(customSituation); }}
              disabled={!customSituation.trim() || generationStatus === 'loading'}
              style={{ marginTop: '8px', padding: '10px', background: generationStatus === 'loading' ? '#374151' : '#7c3aed', color: 'white', border: 'none', borderRadius: '6px', cursor: customSituation.trim() && generationStatus !== 'loading' ? 'pointer' : 'not-allowed', fontSize: '12px', width: '100%', opacity: customSituation.trim() && generationStatus !== 'loading' ? 1 : 0.7 }}
            >
              {generationStatus === 'loading' ? 'Generating...' : 'Generate with AI'}
            </button>

            {generationStatus === 'success' && (
              <div style={{ marginTop: '8px', padding: '8px', background: '#065f46', borderRadius: '6px', fontSize: '12px' }}>
                Positions generated!
              </div>
            )}

            {generationStatus === 'error' && !aiRefusalMessage && (
              <div style={{ marginTop: '8px', padding: '8px', background: '#7f1d1d', borderRadius: '6px', fontSize: '12px' }}>
                Generation failed
              </div>
            )}

            {aiRefusalMessage && (
              <div style={{ marginTop: '8px', padding: '8px', background: '#92400e', borderRadius: '6px', fontSize: '11px' }}>
                Warning: {aiRefusalMessage}
              </div>
            )}
          </div>

          {/* BOX 4: TEAM SETTINGS */}
          <div style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01))', padding: '18px', borderRadius: '14px', border: '1px solid rgba(255,255,255,0.06)', boxShadow: '0 8px 24px rgba(2,6,23,0.6)', backdropFilter: 'blur(6px)', flexShrink: 0 }}>
            <h2 style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', marginBottom: '14px', letterSpacing: '0.05em' }}>
              Team Settings
            </h2>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>
                Appearance and formation for the active view.
              </div>
              <input
                value={teamName}
                onChange={(e) => setTeamName(e.target.value)}
                placeholder="Team Name"
                style={{ width: '100%', background: 'rgba(11,18,36,0.6)', border: '1px solid rgba(255,255,255,0.04)', color: '#fff', padding: '10px 12px', borderRadius: '8px', fontSize: 15, fontWeight: 700, boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.02)'}}
              />

              <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: 42, height: 42, borderRadius: 8, overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0b1224', border: '1px solid rgba(255,255,255,0.04)', boxShadow: '0 4px 10px rgba(11,18,36,0.6)' }}>
                    <input type="color" value={teamColor} onChange={(e) => setTeamColor(e.target.value)} style={{ width: '100%', height: '100%', padding: 0, border: 'none', background: 'transparent', cursor: 'pointer' }} />
                  </div>
                </div>

                <select value={formation} onChange={(e) => {
                  const val = e.target.value;
                  setFormation(val);
                  const preset = FORMATION_PRESETS.find(p => p.name === val);
                  if (preset) handleLoadPreset(preset);
                }} style={{ flex: 1, background: 'rgba(11,18,36,0.5)', border: '1px solid rgba(255,255,255,0.04)', color: '#fff', borderRadius: '8px', padding: '10px 12px', fontSize: 13 }}>
                  {FORMATION_PRESETS.map(p => <option key={p.name} value={p.name}>{p.name}</option>)}
                </select>
              </div>

              
            </div>
          </div>

          {/* BOX 5: PRESET SITUATIONS */}
            <details style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <summary style={{ fontSize: '11px', color: '#6b7280', cursor: 'pointer' }}>Preset Situations</summary>
            <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {SITUATION_PRESETS.map((preset) => (
                <button key={preset.name} onClick={() => handleLoadPreset(preset)} style={{ padding: '8px', background: '#065f46', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontSize: '11px' }}>{preset.name}</button>
              ))}

          {/* BOX 6: TEST ENDPOINT (collapsed) */}
          <details style={{ background: 'rgba(255,255,255,0.03)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <summary style={{ fontSize: '11px', color: '#6b7280', cursor: 'pointer' }}>Debug: Test Endpoint</summary>
            <div style={{ marginTop: '12px' }}>
              <button
                onClick={handleTestEndpoint}
                disabled={testLoading}
                style={{ padding: '8px', background: testLoading ? '#374151' : '#10b981', color: 'white', border: 'none', borderRadius: '6px', cursor: testLoading ? 'not-allowed' : 'pointer', fontSize: '12px', width: '100%' }}
              >
                {testLoading ? 'Testing…' : 'Call /test'}
              </button>
              {testResult && (
                <pre style={{ marginTop: '8px', padding: '8px', background: '#071025', borderRadius: '6px', fontSize: '10px', whiteSpace: 'pre-wrap', overflow: 'auto', maxHeight: '150px' }}>
                  {JSON.stringify(testResult, null, 2)}
                </pre>
              )}
            </div>
          </details>


            </div>
          </details>
          
        </aside>
      )}
    </div>
  );
}


export default App;