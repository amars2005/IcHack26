import { useState, useEffect, useRef } from 'react';
import { Pitch } from './components/Pitch';
import { MenuBar } from './components/MenuBar';
import type { Player } from './types';
import type { Preset } from './presets';
import type { GenerationStatus } from './types';
import { INITIAL_PLAYERS, INITIAL_BALL_CARRIER, PITCH_WIDTH, PITCH_HEIGHT } from './constants';
import { generateSituation, AIError, fetchPlayerMetrics } from './llm';
import { PlayerInfo } from './components/PlayerInfo';

function App() {
  const [players, setPlayers] = useState<Player[]>(INITIAL_PLAYERS);
  const [ballCarrier, setBallCarrier] = useState(INITIAL_BALL_CARRIER);
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus>('idle');
  const [aiRefusalMessage, setAiRefusalMessage] = useState<string | null>(null);
  const [selectedPlayerPosition, setSelectedPlayerPosition] = useState<{ x: number; y: number } | null>(null);
  const [selectedMetrics, setSelectedMetrics] = useState<Record<string, number> | null>(null);
  const [isPitchFullscreen, setIsPitchFullscreen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(true); 
  const pitchContainerRef = useRef<HTMLDivElement | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [xTResult, setXTResult] = useState<any | null>(null);
  const [xTLoading, setXTLoading] = useState(false);
  
  const [teamName, setTeamName] = useState('My Team');
  const [teamColor, setTeamColor] = useState('#ffd43b');
  const [formation, setFormation] = useState('4-4-2');

  const invertHex = (hex: string) => {
    try {
      const cleaned = hex.replace('#', '');
      const num = parseInt(cleaned, 16);
      const inv = (0xffffff ^ num).toString(16).padStart(6, '0');
      return `#${inv}`;
    } catch (e) { return '#3b82f6'; }
  };
  const opponentColor = invertHex(teamColor);

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

  const [viewport, setViewport] = useState({ w: window.innerWidth, h: window.innerHeight });
  useEffect(() => {
    const onResize = () => setViewport({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const sidebarWidth = 380;
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

      {/* PITCH AREA */}
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
        <div style={{ position: 'absolute', left: 40, top: 40, zIndex: 10 }}>
          <h1 style={{ margin: 0, fontSize: 36, fontWeight: 900, color: teamColor }}>{teamName.toUpperCase()}</h1>
        </div>

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
        />
      </main>

      {/* PERSISTENT SIDEBAR */}
      {!isPitchFullscreen && (
        <aside style={{ 
          width: sidebarWidth, 
          height: '100%', 
          background: '#0f172a', 
          borderLeft: '1px solid rgba(255,255,255,0.1)', 
          display: 'flex', 
          flexDirection: 'column',
          padding: '24px',
          gap: '16px',
          overflowY: 'auto', // Allows scrolling if content is too long
          zIndex: 100,
          position: 'relative'
        }}>
          {/* TEAM SETTINGS - Always visible */}
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '16px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <h2 style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', marginBottom: '12px' }}>Team Settings</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <input value={teamName} onChange={(e) => setTeamName(e.target.value)} placeholder="Team Name" style={{ width: '100%', background: '#1e293b', border: '1px solid #334155', color: '#fff', padding: '8px', borderRadius: '6px' }} />
              <div style={{ display: 'flex', gap: '8px' }}>
                <input type="color" value={teamColor} onChange={(e) => setTeamColor(e.target.value)} style={{ width: '40px', height: '36px', border: 'none', background: 'none', cursor: 'pointer' }} />
                <select value={formation} onChange={(e) => setFormation(e.target.value)} style={{ flex: 1, background: '#1e293b', border: '1px solid #334155', color: '#fff', borderRadius: '6px', padding: '0 8px' }}>
                  <option value="4-4-2">4-4-2</option>
                  <option value="4-3-3">4-3-3</option>
                  <option value="3-5-2">3-5-2</option>
                </select>
              </div>
            </div>
          </div>

          {/* TACTICAL MENU moved to overlay (kept in sidebar area intentionally empty) */}

          {/* PLAYER INFO - Always visible at bottom */}
          <div style={{ marginTop: 'auto' }}>
            <PlayerInfo
              playerId={ballCarrier}
              playerPosition={selectedPlayerPosition || undefined}
              metrics={selectedMetrics}
              onClose={undefined}
            />
          </div>
        </aside>
      )}

      {/* MenuBar overlay separate from the sidebar so it doesn't affect right-box layout */}
      {!isPitchFullscreen && !isLoading && (
        <div style={{ position: 'fixed', right: 20, top: 20, width: 300, zIndex: 300 }}>
          <MenuBar
            players={players}
            isOpen={menuOpen}
            ballCarrier={ballCarrier}
            onOpenChange={setMenuOpen}
            onBallCarrierChange={setBallCarrier}
            onLoadPreset={handleLoadPreset}
            onGenerateCustom={handleGenerateCustom}
            generationStatus={generationStatus}
            aiRefusalMessage={aiRefusalMessage}
            teamName={teamName}
            teamColor={teamColor}
            opponentColor={opponentColor}
            onCalculateXT={async () => {
              setXTLoading(true);
              try {
                const attackers = players.filter((p) => p.type === 'attacker').map((p) => ({ x: p.position.x, y: p.position.y, id: p.id, team: 1 }));
                const defenders = players.filter((p) => p.type === 'defender').map((p) => ({ x: p.position.x, y: p.position.y, id: p.id, team: 0 }));
                const keepers = players.filter(p => p.id === '1' || p.id === 'd1').map(p => ({ x: p.position.x, y: p.position.y, id: p.id, team: p.id === '1' ? 1 : 0 }));
                const resp = await fetch('http://localhost:5001/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ attackers, defenders, keepers, ball_id: ballCarrier }) });
                setXTResult(await resp.json());
              } catch (err) { setXTResult({ error: 'xT failed' }); }
              finally { setXTLoading(false); }
            }}
            xTResult={xTResult}
            xTLoading={xTLoading}
          />
        </div>
      )}
    </div>
  );
}

export default App;