import { useState } from 'react';
import type { Player } from '../types';
import type { Preset } from '../presets';
import type { GenerationStatus } from '../types';
import { COLORS } from '../constants';
import { FORMATION_PRESETS, SITUATION_PRESETS } from '../presets';

type MenuBarProps = {
  players: Player[];
  isOpen: boolean;
  ballCarrier: string;
  onOpenChange: (open: boolean) => void;
  onBallCarrierChange: (playerId: string) => void;
  onLoadPreset: (preset: Preset) => void;
  onGenerateCustom: (situation: string) => Promise<void>;
  generationStatus: GenerationStatus;
  aiRefusalMessage?: string | null;
};

export function MenuBar({
  players,
  isOpen,
  ballCarrier,
  onOpenChange,
  onBallCarrierChange,
  onLoadPreset,
  onGenerateCustom,
  generationStatus
  , aiRefusalMessage
}: MenuBarProps) {
  const [customSituation, setCustomSituation] = useState('');
  const attackers = players.filter((p) => p.type === 'attacker');

  const handleGenerate = async () => {
    if (!customSituation.trim()) return;
    await onGenerateCustom(customSituation);
  };

  return (
    <>
      {/* Toggle button - Arrow in top right */}
      <button
        onClick={() => onOpenChange(!isOpen)}
        style={{
          position: 'fixed',
          right: isOpen ? '310px' : '10px',
          top: '10px',
          padding: '10px 12px',
          background: '#1f2937',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '20px',
          transition: 'right 0.3s ease',
          zIndex: 1001,
        }}
      >
        {isOpen ? '→' : '←'}
      </button>

      {/* Menu panel */}
      <div
        style={{
          position: 'fixed',
          right: isOpen ? 0 : '-300px',
          top: 0,
          width: '300px',
          height: '100vh',
          background: '#1f2937',
          color: 'white',
          padding: '20px',
          boxShadow: '-2px 0 10px rgba(0,0,0,0.3)',
          overflowY: 'auto',
          transition: 'right 0.3s ease',
          zIndex: 1000,
        }}
      >
        <div style={{ marginBottom: '24px', marginTop: '10px' }}>
          <h2 style={{ margin: 0, fontSize: '20px' }}>Controls</h2>
        </div>

        {/* Ball possession control */}
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>Ball Possession</h3>
          <label style={{ display: 'block', fontSize: '14px', marginBottom: '6px', color: '#9ca3af' }}>
            Player with ball:
          </label>
          <select
            value={ballCarrier}
            onChange={(e) => onBallCarrierChange(e.target.value)}
            style={{
              width: '100%',
              padding: '8px',
              background: '#374151',
              color: 'white',
              border: '1px solid #4b5563',
              borderRadius: '4px',
              fontSize: '14px',
              cursor: 'pointer',
            }}
          >
            {attackers.map((player) => (
              <option key={player.id} value={player.id}>
                #{player.id}
              </option>
            ))}
          </select>
        </div>

        {/* Player color key */}
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>Team Colors</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '14px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '16px', height: '16px', borderRadius: '50%', background: COLORS.attacker }} />
              <span>Attacking Team</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '16px', height: '16px', borderRadius: '50%', background: COLORS.defender }} />
              <span>Defending Team</span>
            </div>
          </div>
        </div>

        {/* Custom Situation Generator */}
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>Custom Situation</h3>
          <textarea
            value={customSituation}
            onChange={(e) => setCustomSituation(e.target.value)}
            placeholder="Describe an attacking situation (e.g., 'Quick throw-in near the penalty box with numbers up')"
            style={{
              width: '100%',
              padding: '8px',
              background: '#374151',
              color: 'white',
              border: '1px solid #4b5563',
              borderRadius: '4px',
              fontSize: '13px',
              minHeight: '80px',
              resize: 'vertical',
              fontFamily: 'inherit',
            }}
          />
          <button
            onClick={handleGenerate}
            disabled={!customSituation.trim() || generationStatus === 'loading'}
            style={{
              marginTop: '8px',
              padding: '10px',
              background: generationStatus === 'loading' ? '#6b7280' : '#7c3aed',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: customSituation.trim() && generationStatus !== 'loading' ? 'pointer' : 'not-allowed',
              fontSize: '13px',
              width: '100%',
              opacity: customSituation.trim() && generationStatus !== 'loading' ? 1 : 0.6,
            }}
          >
            {generationStatus === 'loading' ? 'Generating...' : 'Generate with AI'}
          </button>

          {/* Status messages */}
          {generationStatus === 'success' && (
            <div style={{
              marginTop: '8px',
              padding: '8px',
              background: '#065f46',
              borderRadius: '4px',
              fontSize: '13px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <span style={{ fontSize: '16px' }}>✓</span>
              <span>Successfully generated positions!</span>
            </div>
          )}

          {generationStatus === 'error' && !aiRefusalMessage && (
            <div style={{
              marginTop: '8px',
              padding: '8px',
              background: '#7f1d1d',
              borderRadius: '4px',
              fontSize: '13px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <span style={{ fontSize: '16px' }}>✗</span>
              <span>Generation failed. Check backend connection.</span>
            </div>
          )}

          {/* AI refusal (inappropriate prompt) message */}
          {aiRefusalMessage && (
            <div style={{
              marginTop: '8px',
              padding: '8px',
              background: '#92400e',
              borderRadius: '4px',
              fontSize: '13px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <span style={{ fontSize: '16px' }}>⚠</span>
              <div>
                <div style={{ fontWeight: '600' }}>AI refused to answer</div>
                <div style={{ fontSize: '12px', color: '#fde68a' }}>{aiRefusalMessage}</div>
              </div>
            </div>
          )}
        </div>

        {/* Formation Presets */}
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>Formations</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
            {FORMATION_PRESETS.map((preset) => (
              <button
                key={preset.name}
                onClick={() => onLoadPreset(preset)}
                style={{
                  padding: '10px',
                  background: '#374151',
                  color: 'white',
                  border: '1px solid #4b5563',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '13px',
                }}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>

        {/* Attacking Situations */}
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>Attacking Situations</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {SITUATION_PRESETS.map((preset) => (
              <button
                key={preset.name}
                onClick={() => onLoadPreset(preset)}
                style={{
                  padding: '10px',
                  background: '#065f46',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '13px',
                }}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>


        {/* Instructions */}
        <div style={{ fontSize: '12px', color: '#9ca3af', lineHeight: '1.6' }}>
          <p style={{ margin: '0 0 8px' }}>Drag players to reposition them on the pitch.</p>
          <p style={{ margin: '0 0 8px' }}>Yellow ring indicates ball possession.</p>
          <p style={{ margin: '0 0 8px' }}>AI generation powered by backend server.</p>
        </div>
      </div>
    </>
  );
}
