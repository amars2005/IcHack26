import { useState } from 'react';
import { Pitch } from './components/Pitch';
import { MenuBar } from './components/MenuBar';
import type { Player } from './types';
import type { Preset } from './presets';
import type { GenerationStatus, XTResult } from './types';
import { INITIAL_PLAYERS, INITIAL_BALL_CARRIER, PITCH_WIDTH, PITCH_HEIGHT } from './constants';
import { generateSituation } from './llm';
import { usePitchState } from './hooks';

function App() {
  const [players, setPlayers] = useState<Player[]>(INITIAL_PLAYERS);
  const [menuOpen, setMenuOpen] = useState(false);
  const [ballCarrier, setBallCarrier] = useState(INITIAL_BALL_CARRIER);
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus>('idle');
  const [xTResult, setXTResult] = useState<XTResult | null>(null);
  const [xTLoading, setXTLoading] = useState(false);
  const {
    pitchState,
    updatePlayerPosition,
    setBallId,
    setPositions,
    resetPositions,
    getApiPayload,
  } = usePitchState();

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

      // Clear success message after 3 seconds
      setTimeout(() => {
        setGenerationStatus('idle');
      }, 3000);
    } catch (error) {
      console.error('Generation error:', error);
      setGenerationStatus('error');

      // Clear error message after 5 seconds
      setTimeout(() => {
        setGenerationStatus('idle');
      }, 5000);
    }
  };

  // Calculate pitch scale based on menu state - fill 70% of viewport
  const availableWidth = (menuOpen ? window.innerWidth - 320 : window.innerWidth) * 0.7;
  const availableHeight = window.innerHeight * 0.7;
  const scale = Math.min(
    availableWidth / PITCH_WIDTH,
    availableHeight / PITCH_HEIGHT
  );

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
    } catch (error) {
      console.error('xT calculation error:', error);
      setXTResult({ error: 'Failed to calculate xT' });
    } finally {
      setXTLoading(false);
    }
  };

  const handleGeneratePositions = async (situation: string) => {
    const response = await fetch(
      `http://localhost:5001/generate-positions?situation=${encodeURIComponent(situation)}`
    );
    const data = await response.json();
    if (data.attackers && data.defenders && data.ball_id) {
      setPositions(data.attackers, data.defenders, data.ball_id);
    }
  };

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        background: '#0f172a',
        transition: 'all 0.3s ease',
        marginLeft: menuOpen ? '-150px' : '0',
      }}
    >
      <Pitch
        players={players}
        ballCarrier={ballCarrier}
        onPlayerMove={handlePlayerMove}
        scale={scale}
      />
      <MenuBar
        players={players}
        isOpen={menuOpen}
        ballCarrier={ballCarrier}
        onOpenChange={setMenuOpen}
        onBallCarrierChange={setBallCarrier}
        onLoadPreset={handleLoadPreset}
        onGenerateCustom={handleGenerateCustom}
        generationStatus={generationStatus}
        onCalculateXT={handleCalculateXT}
        xTResult={xTResult}
        xTLoading={xTLoading}
      />
    </div>
  );
}

export default App;
