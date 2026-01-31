import { useState } from 'react';
import { Pitch } from './components/Pitch';
import { MenuBar } from './components/MenuBar';
import type { Player } from './types';
import type { Preset } from './presets';
import type { GenerationStatus } from './types';
import { INITIAL_PLAYERS, INITIAL_BALL_CARRIER, PITCH_WIDTH, PITCH_HEIGHT } from './constants';
import { generateSituation } from './llm';

function App() {
  const [players, setPlayers] = useState<Player[]>(INITIAL_PLAYERS);
  const [menuOpen, setMenuOpen] = useState(false);
  const [ballCarrier, setBallCarrier] = useState(INITIAL_BALL_CARRIER);
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus>('idle');

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
      />
    </div>
  );
}

export default App;
