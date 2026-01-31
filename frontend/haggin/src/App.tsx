import { useState, useEffect } from 'react';
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
  // Slightly reduce scale so UI is a little smaller than before
  const baseScale = Math.min(availableWidth / PITCH_WIDTH, availableHeight / PITCH_HEIGHT);
  // Estimate additional vertical space required for the info box + container padding
  const infoBoxEstimatedHeight = 120; // conservative estimate for PlayerInfo height
  const containerPaddingY = 24 * 2; // top + bottom padding on the centered container
  // Compute max scale so pitch + info box fit vertically without scrolling
  const maxScaleByHeight = Math.max(0.35, (availableHeight - infoBoxEstimatedHeight - containerPaddingY) / PITCH_HEIGHT);
  const scale = Math.max(0.35, Math.min(baseScale * 0.92, maxScaleByHeight));
  const pitchPixelWidth = PITCH_WIDTH * scale;
  // Add a small inset so the pitch has a bezel around it
  const displayPitchWidth = Math.max(220, pitchPixelWidth * 0.94);
  // Ensure info box and displayed pitch never touch screen edges or overlap the menu
  const sideMargin = 40; // desired bezel on each side
  const reservedForMenu = menuOpen ? (300 + 20) : 0; // menu width + gap when open
  const maxAvailableWidth = Math.max(180, viewport.w - reservedForMenu - sideMargin * 2);
  const displayPitchWidthClamped = Math.min(displayPitchWidth, maxAvailableWidth);
  const infoBoxWidth = Math.min(Math.max(200, displayPitchWidthClamped * 0.75), maxAvailableWidth);

  // Vertical bezel so pitch/info don't touch top/bottom
  const verticalMargin = 40;
  const maxAvailableHeight = Math.max(200, viewport.h - verticalMargin * 2);
  // No scroll: ensure container fits
  const containerMaxHeight = maxAvailableHeight;

  // Amount to shift the centered pitch left when the menu opens so it remains visible
  const menuShift = menuOpen ? menuWidth / 2 + 12 : 0; // 12px extra padding
  const centerTransform = `translate(calc(-50% - ${menuShift}px), -50%)`;

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        position: 'relative',
        background: '#0f172a',
        transition: 'all 0.3s ease',
      }}
    >
      {/* Centered pitch independent of MenuBar; shifts left when menu opens */}
      <div style={{ position: 'absolute', left: '50%', top: '50%', transform: centerTransform, display: 'flex', flexDirection: 'column', alignItems: 'center', padding: 24, borderRadius: 10, boxSizing: 'border-box', border: '1px solid rgba(255,255,255,0.03)', transition: 'transform 350ms cubic-bezier(0.22,1,0.36,1)', willChange: 'transform' }}>
        <div style={{ width: displayPitchWidthClamped, display: 'flex', justifyContent: 'center' }}>
          <Pitch
            players={players}
            ballCarrier={ballCarrier}
            onPlayerMove={handlePlayerMove}
            scale={scale}
          />
        </div>
        <div style={{ width: infoBoxWidth, marginTop: 16 }}>
          <PlayerInfo
            playerId={ballCarrier}
            playerPosition={selectedPlayerPosition || undefined}
            metrics={selectedMetrics}
            onClose={undefined}
          />
        </div>
      </div>

      {/* MenuBar at top-right as overlay */}
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
        />
      </div>
    </div>
  );
}

export default App;
