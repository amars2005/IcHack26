import { useState } from 'react';
import { Pitch } from './components/Pitch';
import { MenuBar } from './components/MenuBar';
import type { Player, HeatmapData, GenerationStatus } from './types';
import type { Preset } from './presets';
import { INITIAL_PLAYERS, INITIAL_BALL_CARRIER, PITCH_WIDTH, PITCH_HEIGHT } from './constants';
import { generateSituation, AIError } from './llm';

function App() {
  const [players, setPlayers] = useState<Player[]>(INITIAL_PLAYERS);
  const [ballCarrier, setBallCarrier] = useState(INITIAL_BALL_CARRIER);
  const [generationStatus, setGenerationStatus] = useState<GenerationStatus>('idle');
  const [aiRefusalMessage, setAiRefusalMessage] = useState<string | null>(null);
  const [xTResult, setXTResult] = useState<any | null>(null);
  const [xTLoading, setXTLoading] = useState(false);
  const [testResult, setTestResult] = useState<any | null>(null);
  const [testLoading, setTestLoading] = useState(false);
  const [passTestResult, setPassTestResult] = useState<any | null>(null);
  const [passTestLoading, setPassTestLoading] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [heatmapData, setHeatmapData] = useState<HeatmapData | null>(null);
  const [menuOpen, setMenuOpen] = useState(true);

  // Calculate pitch scale based on menu state - fill 70% of viewport
  const availableWidth = (menuOpen ? window.innerWidth - 320 : window.innerWidth) * 0.7;
  const availableHeight = window.innerHeight * 0.7;
  const scale = Math.min(
    availableWidth / PITCH_WIDTH,
    availableHeight / PITCH_HEIGHT
  );

  // Build API payload directly from current players and ballCarrier state
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
    const clampedX = Math.max(0, Math.min(PITCH_WIDTH, x));
    const clampedY = Math.max(0, Math.min(PITCH_HEIGHT, y));
    setPlayers((prev) => prev.map((p) => (p.id === id ? { ...p, position: { x: clampedX, y: clampedY } } : p)));
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
      setTimeout(() => setGenerationStatus('idle'), 3000);
    } catch (error) {
      setGenerationStatus('error');
      if (error instanceof AIError && error.code === 'INAPPROPRIATE_PROMPT') setAiRefusalMessage(error.message);
      setTimeout(() => { setGenerationStatus('idle'); setAiRefusalMessage(null); }, 5000);
    }
  };

  const handleTest = async () => {
    setTestLoading(true);
    setTestResult(null);
    try {
      const payload = getApiPayload();
      console.log('Sending test payload:', payload);
      const response = await fetch('http://localhost:5001/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      console.log('Test Result:', result);
      setTestResult(result);
    } catch (error) {
      console.error('Test error:', error);
      setTestResult({ error: 'Failed to call test endpoint' });
    } finally {
      setTestLoading(false);
    }
  };

  const handleTestPass = async () => {
    setPassTestLoading(true);
    setPassTestResult(null);
    try {
      const payload = getApiPayload();
      console.log('Sending pass test payload:', payload);
      const response = await fetch('http://localhost:5001/test-pass', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      console.log('Pass Test Result:', result);
      setPassTestResult(result);
    } catch (error) {
      console.error('Pass test error:', error);
      setPassTestResult({ error: 'Failed to call test-pass endpoint' });
    } finally {
      setPassTestLoading(false);
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
        aiRefusalMessage={aiRefusalMessage}
        onTest={handleTest}
        testResult={testResult}
        testLoading={testLoading}
        onTestPass={handleTestPass}
        passTestResult={passTestResult}
        passTestLoading={passTestLoading}
      />
    </div>
  );
}

export default App;
