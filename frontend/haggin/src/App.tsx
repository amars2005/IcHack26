import { useState } from 'react';
import { Pitch } from './components/Pitch';
import { MenuBar } from './components/MenuBar';
import type { Player, HeatmapData } from './types';
import type { Preset } from './presets';
import type { GenerationStatus, XTResult } from './types';
import { INITIAL_PLAYERS, INITIAL_BALL_CARRIER, PITCH_WIDTH, PITCH_HEIGHT } from './constants';
import { generateSituation } from './llm';

function App() {
  const [players, setPlayers] = useState<Player[]>(INITIAL_PLAYERS);
  const [menuOpen, setMenuOpen] = useState(false);
  const [ballCarrier, setBallCarrier] = useState(INITIAL_BALL_CARRIER);
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus>('idle');
  const [xTResult, setXTResult] = useState<XTResult | null>(null);
  const [xTLoading, setXTLoading] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapData, setHeatmapData] = useState<HeatmapData | null>(null);

  // Build API payload directly from current players and ballCarrier state
  // Note: SVG has y=0 at top, but football coordinates have y=0 at bottom
  // So we flip y: actualY = PITCH_HEIGHT - svgY
  const getApiPayload = () => {
    const attackers = players
      .filter(p => p.type === 'attacker' && p.id !== '1')
      .map(p => ({ id: p.id, x: p.position.x, y: PITCH_HEIGHT - p.position.y, team: 1 }));
    
    const defenders = players
      .filter(p => p.type === 'defender' && p.id !== 'd1')
      .map(p => ({ id: p.id, x: p.position.x, y: PITCH_HEIGHT - p.position.y, team: 0 }));
    
    const attackerKeeper = players.find(p => p.type === 'attacker' && p.id === '1');
    const defenderKeeper = players.find(p => p.type === 'defender' && p.id === 'd1');
    
    const keepers = [
      attackerKeeper 
        ? { id: attackerKeeper.id, x: attackerKeeper.position.x, y: PITCH_HEIGHT - attackerKeeper.position.y, team: 1 }
        : { id: '1', x: 12, y: 40, team: 1 },
      defenderKeeper
        ? { id: defenderKeeper.id, x: defenderKeeper.position.x, y: PITCH_HEIGHT - defenderKeeper.position.y, team: 0 }
        : { id: 'd1', x: 108, y: 40, team: 0 },
    ];

    return {
      attackers,
      defenders,
      keepers,
      ball_id: ballCarrier,
    };
  };

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

  const handleGeneratePositions = async (situation: string) => {
    const response = await fetch(
      `http://localhost:5001/generate-positions?situation=${encodeURIComponent(situation)}`
    );
    const data = await response.json();
    if (data.attackers && data.defenders && data.ball_id) {
      // Convert API response to players format
      const newAttackers = data.attackers.map((a: { id: string; x: number; y: number }) => ({
        id: a.id,
        type: 'attacker' as const,
        position: { x: a.x, y: a.y },
      }));
      const newDefenders = data.defenders.map((d: { id: string; x: number; y: number }) => ({
        id: d.id,
        type: 'defender' as const,
        position: { x: d.x, y: d.y },
      }));
      setPlayers([...newAttackers, ...newDefenders]);
      setBallCarrier(data.ball_id);
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
        heatmap={heatmapData}
        showHeatmap={showHeatmap}
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
        showHeatmap={showHeatmap}
        onToggleHeatmap={() => setShowHeatmap(!showHeatmap)}
        hasHeatmapData={!!heatmapData}
      />
    </div>
  );
}

export default App;
