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
  const [menuOpen, setMenuOpen] = useState(false);
  const [ballCarrier, setBallCarrier] = useState(INITIAL_BALL_CARRIER);
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus>('idle');
  const [aiRefusalMessage, setAiRefusalMessage] = useState<string | null>(null);
  const [selectedPlayerPosition, setSelectedPlayerPosition] = useState<{ x: number; y: number } | null>(null);
  const [selectedMetrics, setSelectedMetrics] = useState<Record<string, number> | null>(null);
  const [isPitchFullscreen, setIsPitchFullscreen] = useState(false);
  const pitchContainerRef = useRef<HTMLDivElement | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [xTResult, setXTResult] = useState<any | null>(null);
  const [xTLoading, setXTLoading] = useState(false);
  // Prescreen state shown after the initial loader
  const [prescreenVisible, setPrescreenVisible] = useState(true);
  const [teamName, setTeamName] = useState('My Team');
  const [teamColor, setTeamColor] = useState('#ffd43b');
  const [formation, setFormation] = useState('4-4-2');
  // Compute opponent color as the hex inverse of the chosen team color
  const invertHex = (hex: string) => {
    try {
      const cleaned = hex.replace('#', '');
      const num = parseInt(cleaned, 16);
      const inv = (0xffffff ^ num).toString(16).padStart(6, '0');
      return `#${inv}`;
    } catch (e) {
      return '#3b82f6';
    }
  };
  const opponentColor = invertHex(teamColor);

  const handlePlayerMove = (id: string, x: number, y: number) => {
    // Clamp position to pitch boundaries
    const clampedX = Math.max(0, Math.min(PITCH_WIDTH, x));
    const clampedY = Math.max(0, Math.min(PITCH_HEIGHT, y));

    setPlayers((prev) =>
      prev.map((p) =>
        p.id === id ? { ...p, position: { x: clampedX, y: clampedY } } : p
      )
    );
  };

  // Always show info for the current ball carrier. Fetch metrics when ball carrier or players change.
  useEffect(() => {
    const id = ballCarrier;
    const p = players.find((pl) => pl.id === id);
    setSelectedPlayerPosition(p ? p.position : null);
    setSelectedMetrics(null);
    if (!id) return;
    let mounted = true;
    fetchPlayerMetrics(id).then((metrics) => {
      if (mounted) setSelectedMetrics(metrics);
    }).catch(() => {
      if (mounted) setSelectedMetrics(null);
    });
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
      setAiRefusalMessage(null);

      // Clear success message after 3 seconds
      setTimeout(() => {
        setGenerationStatus('idle');
      }, 3000);
    } catch (error) {
      console.error('Generation error:', error);
      // If backend explicitly refused due to inappropriate prompt, show a helpful message
      if (error instanceof AIError && error.code === 'INAPPROPRIATE_PROMPT') {
        setAiRefusalMessage(error.message || 'Prompt rejected by policy');
      } else {
        setAiRefusalMessage(null);
      }
      setGenerationStatus('error');

      // Clear error message after 5 seconds
      setTimeout(() => {
        setGenerationStatus('idle');
        setAiRefusalMessage(null);
      }, 5000);
    }
  };

  // Responsive layout: track viewport to recompute scale on resize
  const [viewport, setViewport] = useState({ w: typeof window !== 'undefined' ? window.innerWidth : 1200, h: typeof window !== 'undefined' ? window.innerHeight : 800 });
  useEffect(() => {
    const onResize = () => setViewport({ w: window.innerWidth, h: window.innerHeight });
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const menuWidth = menuOpen ? 300 : 0;
  // Give the pitch more screen real-estate (90% of remaining area), but constrain by height as well
  const availableWidth = Math.max(200, (viewport.w - menuWidth) * 0.9);
  const availableHeight = Math.max(200, viewport.h * 0.9);

  // If in fullscreen, compute scale to use almost the full viewport (ignore menu/info)
  let scale: number;
  if (typeof window !== 'undefined' && isPitchFullscreen) {
    const fsPadding = 24; // small inset inside fullscreen
    const fsAvailableWidth = Math.max(100, viewport.w - fsPadding * 2);
    const fsAvailableHeight = Math.max(100, viewport.h - fsPadding * 2);
    scale = Math.max(0.2, Math.min(fsAvailableWidth / PITCH_WIDTH, fsAvailableHeight / PITCH_HEIGHT));
  } else {
    // Slightly reduce scale so UI is a little smaller than before
    const baseScale = Math.min(availableWidth / PITCH_WIDTH, availableHeight / PITCH_HEIGHT);
    // Estimate additional vertical space required for the info box + container padding
    const infoBoxEstimatedHeight = 120; // conservative estimate for PlayerInfo height
    const containerPaddingY = 24 * 2; // top + bottom padding on the centered container
    // Compute max scale so pitch + info box fit vertically without scrolling
    const maxScaleByHeight = Math.max(0.35, (availableHeight - infoBoxEstimatedHeight - containerPaddingY) / PITCH_HEIGHT);
    scale = Math.max(0.35, Math.min(baseScale * 0.92, maxScaleByHeight));
  }
  const pitchPixelWidth = PITCH_WIDTH * scale;
  // Add a small inset so the pitch has a bezel around it
  const displayPitchWidth = Math.max(220, pitchPixelWidth * 0.94);
  // Ensure info box and displayed pitch never touch screen edges or overlap the menu
  const sideMargin = 40; // desired bezel on each side
  const reservedForMenu = menuOpen ? (300 + 20) : 0; // menu width + gap when open
  const maxAvailableWidth = Math.max(180, viewport.w - reservedForMenu - sideMargin * 2);
  const displayPitchWidthClamped = Math.min(displayPitchWidth, maxAvailableWidth);
  const infoBoxWidth = Math.min(Math.max(200, displayPitchWidthClamped * 0.75), maxAvailableWidth);

  // Vertical bezel handled via scale and padding

  // Amount to shift the centered pitch left when the menu opens so it remains visible
  const menuShift = menuOpen ? menuWidth / 2 + 12 : 0; // 12px extra padding
  const centerTransform = `translate(calc(-50% - ${menuShift}px), -50%)`;

  // Keep fullscreen state in sync with the browser Fullscreen API
  useEffect(() => {
    const onFsChange = () => {
      setIsPitchFullscreen(!!document.fullscreenElement && document.fullscreenElement === pitchContainerRef.current);
    };
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, []);

  // Startup loading screen for 2 seconds
  useEffect(() => {
    const t = setTimeout(() => setIsLoading(false), 2000);
    return () => clearTimeout(t);
  }, []);

  const togglePitchFullscreen = async () => {
    try {
      if (!document.fullscreenElement) {
        if (pitchContainerRef.current) await pitchContainerRef.current.requestFullscreen();
      } else {
        if (document.exitFullscreen) await document.exitFullscreen();
      }
    } catch (err) {
      console.error('Failed to toggle fullscreen', err);
    }
  };

return (
    <div style={{position: 'relative', minHeight: '100vh', background: '#071126', color: '#dbeafe'}}>
      {isLoading && (
  <div style={{
    position: 'fixed',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#071126',
    zIndex: 10000,
    flexDirection: 'column',
    gap: 20
  }} aria-live="polite">
    <style>{`
      @keyframes roll {
        0% { transform: translateX(-28px) rotate(0deg); }
        50% { transform: translateX(28px) rotate(-360deg); }
        100% { transform: translateX(-28px) rotate(-720deg); }
      }
      @keyframes bounce { 0%,100%{ transform: translateY(0); } 50%{ transform: translateY(-10px); } }
    `}</style>
    
    <div style={{
      width: 110,
      height: 110,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      animation: 'bounce 1.2s ease-in-out infinite'
    }}>
      <div style={{ animation: 'roll 1.6s linear infinite', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <img src="/football.svg" alt="football" style={{ width: 104, height: 104, display: 'block', transformOrigin: '50% 50%' }} />
      </div>
    </div>
    
    <div style={{
      color: '#dbeafe',
      fontSize: 18,
      fontWeight: 600,
      animation: 'pulse 1.5s ease-in-out infinite'
    }}>
      Loading...
    </div>
  </div>
)}
      {/* Prescreen: select team details after loader */}
      {!isLoading && prescreenVisible && (
        <div style={{ position: 'fixed', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(7,17,38,0.9)', zIndex: 10001 }}>
          <div style={{ width: 520, maxWidth: '92%', background: '#0b1224', borderRadius: 12, padding: 20, boxShadow: '0 8px 30px rgba(2,6,23,0.6)', color: '#e6eef8' }}>
            <h2 style={{ margin: 0, marginBottom: 12 }}>Team setup</h2>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <div style={{ flex: 1 }}>
                <label style={{ display: 'block', fontSize: 13, marginBottom: 6 }}>Team name</label>
                <input value={teamName} onChange={(e) => setTeamName(e.target.value)} style={{ width: '100%', padding: '8px 10px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.08)', background: '#071124', color: '#e6eef8' }} />
              </div>
              <div style={{ width: 120 }}>
                <label style={{ display: 'block', fontSize: 13, marginBottom: 6 }}>Team color</label>
                <input type="color" value={teamColor} onChange={(e) => setTeamColor(e.target.value)} style={{ width: '100%', height: 40, padding: 0, border: 'none', background: 'transparent' }} />
              </div>
            </div>
            <div style={{ marginTop: 12 }}>
              <label style={{ display: 'block', fontSize: 13, marginBottom: 6 }}>Formation</label>
              <select value={formation} onChange={(e) => setFormation(e.target.value)} style={{ width: '100%', padding: '8px 10px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.08)', background: '#071124', color: '#e6eef8' }}>
                <option value="4-4-2">4-4-2</option>
                <option value="4-3-3">4-3-3</option>
                <option value="3-5-2">3-5-2</option>
                <option value="5-3-2">5-3-2</option>
                <option value="4-2-3-1">4-2-3-1</option>
              </select>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
              <button onClick={() => setPrescreenVisible(false)} style={{ padding: '8px 12px', borderRadius: 8, background: 'rgba(255,255,255,0.06)', color: '#e6eef8', border: 'none', cursor: 'pointer' }}>Skip</button>
              <button onClick={() => { setPrescreenVisible(false); /* optionally apply formation to players later */ }} style={{ padding: '8px 12px', borderRadius: 8, background: teamColor, color: '#071124', border: 'none', cursor: 'pointer', fontWeight: 700 }}>Apply</button>
            </div>
          </div>
        </div>
      )}
      {/* Centered pitch independent of MenuBar; shifts left when menu opens */}
      <div
        ref={pitchContainerRef}
        style={isPitchFullscreen ? {
          position: 'fixed',
          left: 0,
          top: 0,
          width: '100vw',
          height: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#000',
          padding: 12,
          boxSizing: 'border-box',
          zIndex: 9999
        } : {
          position: 'absolute',
          left: '50%',
          top: '50%',
          transform: centerTransform,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: 24,
          borderRadius: 10,
          boxSizing: 'border-box',
          border: '1px solid rgba(255,255,255,0.03)',
          transition: 'transform 350ms cubic-bezier(0.22,1,0.36,1)',
          willChange: 'transform'
        }}
      >
        {/* fullscreen toggle button */}
        <button
          onClick={togglePitchFullscreen}
          aria-label={isPitchFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
          title={isPitchFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
          style={{ position: 'absolute', right: 12, top: 12, zIndex: 50, background: 'rgba(0,0,0,0.45)', color: '#fff', border: 'none', padding: '6px 8px', borderRadius: 6, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        >
          {isPitchFullscreen ? (
            // Exit fullscreen icon (inward corners)
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
              <path d="M9 3H5a2 2 0 0 0-2 2v4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M15 3h4a2 2 0 0 1 2 2v4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M9 21H5a2 2 0 0 1-2-2v-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M15 21h4a2 2 0 0 0 2-2v-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : (
            // Enter fullscreen icon (outward corners)
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
              <path d="M3 9V5a2 2 0 0 1 2-2h4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M21 9V5a2 2 0 0 0-2-2h-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M3 15v4a2 2 0 0 0 2 2h4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M21 15v4a2 2 0 0 1-2 2h-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </button>
        {/* Team name (left-justified) */}
        <div style={{ position: 'absolute', left: 24, top: 12, zIndex: 60, pointerEvents: 'none', display: 'flex', alignItems: 'center' }}>
          <div style={{ background: 'transparent', padding: '6px 10px', borderRadius: 6, color: '#ffffff', fontWeight: 700, fontSize: isPitchFullscreen ? 26 : 18, letterSpacing: 0.4, boxShadow: 'none', fontFamily: 'Helvetica, Arial, sans-serif', textTransform: 'none' }}>{teamName}</div>
        </div>
        <div style={{ width: isPitchFullscreen ? Math.min(viewport.w - 24, displayPitchWidthClamped) : displayPitchWidthClamped, display: 'flex', justifyContent: 'center' }}>
          <Pitch
            players={players}
            ballCarrier={ballCarrier}
            onPlayerMove={handlePlayerMove}
            scale={scale}
            teamColor={teamColor}
          />
        </div>
        {!isPitchFullscreen && (
          <div style={{ width: infoBoxWidth, marginTop: 16 }}>
            <PlayerInfo
              playerId={ballCarrier}
              playerPosition={selectedPlayerPosition || undefined}
              metrics={selectedMetrics}
              onClose={undefined}
            />
          </div>
        )}
      </div>

      {/* MenuBar at top-right as overlay */}
      {!isPitchFullscreen && (
        <div style={{ position: 'absolute', right: 20, top: 20, width: 300 }}>
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
              setXTResult(null);
              try {
                // Build payload from current players array
                const attackers = players.filter((p) => p.type === 'attacker').map((p) => ({ x: p.position.x, y: p.position.y, id: p.id, team: 1 }));
                const defenders = players.filter((p) => p.type === 'defender').map((p) => ({ x: p.position.x, y: p.position.y, id: p.id, team: 0 }));
                // Keepers: attacker keeper id '1' then defender keeper id 'd1'
                const keeperAtt = players.find((p) => p.id === '1');
                const keeperDef = players.find((p) => p.id === 'd1');
                const keepers = [] as Array<{ x: number; y: number; id: string; team: number }>;
                if (keeperAtt) keepers.push({ x: keeperAtt.position.x, y: keeperAtt.position.y, id: keeperAtt.id, team: 1 });
                if (keeperDef) keepers.push({ x: keeperDef.position.x, y: keeperDef.position.y, id: keeperDef.id, team: 0 });

                const payload = {
                  attackers,
                  defenders,
                  keepers,
                  ball_id: ballCarrier,
                };

                const resp = await fetch('http://localhost:5001/', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                const data = await resp.json();
                setXTResult(data);
              } catch (err) {
                console.error('xT calc error', err);
                setXTResult({ error: 'xT failed' } as any);
              } finally {
                setXTLoading(false);
              }
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
