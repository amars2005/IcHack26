import { useState } from 'react';
import type { Player } from '../types';
import type { Preset } from '../presets';
import type { GenerationStatus, XTResult } from '../types';
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
  onCalculateXT: () => Promise<void>;
  xTResult: XTResult | null;
  xTLoading: boolean;
  showHeatmap: boolean;
  onToggleHeatmap: () => void;
  hasHeatmapData: boolean;
  aiRefusalMessage?: string | null;
  teamName?: string;
  teamColor?: string;
  opponentColor?: string;
  onTest?: () => Promise<void>;
  testResult?: any | null;
  testLoading?: boolean;
  onTestPass?: () => Promise<void>;
  passTestResult?: any | null;
  passTestLoading?: boolean;
};

export function MenuBar({
  players,
  isOpen,
  ballCarrier,
  onOpenChange,
  onBallCarrierChange,
  onLoadPreset,
  onGenerateCustom,
  generationStatus,
  onCalculateXT,
  xTResult,
  xTLoading,
  showHeatmap,
  onToggleHeatmap,
  hasHeatmapData,
  aiRefusalMessage,
  teamName,
  teamColor,
  opponentColor,
  onTest,
  testResult,
  testLoading,
  onTestPass,
  passTestResult,
  passTestLoading,
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
              <div style={{ width: '16px', height: '16px', borderRadius: '50%', background: teamColor || COLORS.attacker }} />
              <span>{teamName || 'Attacking Team'}</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div style={{ width: '16px', height: '16px', borderRadius: '50%', background: opponentColor || COLORS.defender }} />
              <span>{opponentColor ? 'Opposition' : 'Defending Team'}</span>
            </div>
          </div>
        </div>

        {/* Calculate Actions Section */}
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>Action Analysis</h3>
          <p style={{ fontSize: '13px', color: '#9ca3af', marginTop: 0, marginBottom: '12px' }}>
            Analyze possible actions for the ball carrier.
          </p>
          <button
            onClick={async () => { try { await onCalculateXT(); } catch (e) { console.error(e); } }}
            disabled={xTLoading}
            style={{
              padding: '12px',
              background: xTLoading ? '#6b7280' : '#f59e0b',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: xTLoading ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: 'bold',
              width: '100%',
              opacity: xTLoading ? 0.6 : 1,
            }}
          >
            {xTLoading ? 'Analyzing…' : '⚡ Analyze Actions'}
          </button>

          {xTResult && !xTResult.error && (
            <div style={{ marginTop: '12px', background: '#0b1224', borderRadius: '8px', overflow: 'hidden' }}>
              {/* Shoot action */}
              {xTResult.shoot && (
                <div style={{ 
                  padding: '12px', 
                  borderBottom: '1px solid #1e293b',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '18px' }}>⚽</span>
                    <span style={{ fontWeight: 600, color: '#f59e0b' }}>Shoot</span>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#22c55e' }}>
                      {(xTResult.shoot.xG * 100).toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '10px', color: '#9ca3af' }}>xG</div>
                  </div>
                </div>
              )}

              {/* Carry action */}
              {xTResult.carry && xTResult.carry.xT !== null && (
                <div style={{ 
                  padding: '12px', 
                  borderBottom: '1px solid #1e293b',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '18px' }}>🏃</span>
                    <span style={{ fontWeight: 600, color: '#3b82f6' }}>Carry</span>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#22c55e' }}>
                      {typeof xTResult.carry.xT === 'number' ? xTResult.carry.xT.toFixed(3) : '—'}
                    </div>
                    <div style={{ fontSize: '10px', color: '#9ca3af' }}>xT</div>
                  </div>
                </div>
              )}

              {/* Pass actions - sorted by success probability */}
              {(() => {
                const passActions = Object.entries(xTResult)
                  .filter(([key]) => key.startsWith('pass_to_'))
                  .map(([key, value]: [string, any]) => ({
                    playerId: key.replace('pass_to_', ''),
                    xT: value.xT,
                    probability: value['P(success)']
                  }))
                  .sort((a, b) => (b.probability ?? 0) - (a.probability ?? 0));

                if (passActions.length === 0) return null;

                return (
                  <div style={{ padding: '12px' }}>
                    <div style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '8px', 
                      marginBottom: '10px',
                      paddingBottom: '8px',
                      borderBottom: '1px solid #374151'
                    }}>
                      <span style={{ fontSize: '18px' }}>📤</span>
                      <span style={{ fontWeight: 600, color: '#a855f7' }}>Pass Options</span>
                    </div>
                    <div style={{ 
                      display: 'grid', 
                      gridTemplateColumns: 'auto 1fr auto', 
                      gap: '6px 12px',
                      fontSize: '13px'
                    }}>
                      <div style={{ color: '#9ca3af', fontSize: '11px', fontWeight: 600 }}>TO</div>
                      <div style={{ color: '#9ca3af', fontSize: '11px', fontWeight: 600 }}>SUCCESS</div>
                      <div style={{ color: '#9ca3af', fontSize: '11px', fontWeight: 600, textAlign: 'right' }}>xT</div>
                      {passActions.map(({ playerId, xT, probability }) => (
                        <>
                          <div key={`player-${playerId}`} style={{ fontWeight: 500 }}>#{playerId}</div>
                          <div key={`prob-${playerId}`}>
                            <div style={{ 
                              display: 'flex', 
                              alignItems: 'center', 
                              gap: '6px' 
                            }}>
                              <div style={{ 
                                flex: 1, 
                                height: '6px', 
                                background: '#374151', 
                                borderRadius: '3px',
                                overflow: 'hidden'
                              }}>
                                <div style={{ 
                                  width: `${(probability ?? 0) * 100}%`, 
                                  height: '100%', 
                                  background: probability >= 0.7 ? '#22c55e' : probability >= 0.4 ? '#f59e0b' : '#ef4444',
                                  borderRadius: '3px'
                                }} />
                              </div>
                              <span style={{ 
                                fontSize: '11px', 
                                color: probability >= 0.7 ? '#22c55e' : probability >= 0.4 ? '#f59e0b' : '#ef4444',
                                minWidth: '36px',
                                textAlign: 'right'
                              }}>
                                {probability !== null ? `${(probability * 100).toFixed(0)}%` : '—'}
                              </span>
                            </div>
                          </div>
                          <div key={`xt-${playerId}`} style={{ textAlign: 'right', color: '#9ca3af' }}>
                            {xT !== null ? xT.toFixed(3) : '—'}
                          </div>
                        </>
                      ))}
                    </div>
                  </div>
                );
              })()}
            </div>
          )}

          {xTResult?.error && (
            <div style={{
              marginTop: '12px',
              padding: '12px',
              background: '#7f1d1d',
              borderRadius: '4px',
              fontSize: '13px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <span style={{ fontSize: '16px' }}>✗</span>
              <span>{xTResult.error}</span>
            </div>
          )}

          {/* Heatmap toggle */}
          {hasHeatmapData && (
            <div style={{ marginTop: '12px' }}>
              <label style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '10px',
                cursor: 'pointer',
                fontSize: '14px',
              }}>
                <input
                  type="checkbox"
                  checked={showHeatmap}
                  onChange={onToggleHeatmap}
                  style={{ 
                    width: '18px', 
                    height: '18px',
                    cursor: 'pointer',
                  }}
                />
                Show xT Heatmap
              </label>
              <div style={{ 
                marginTop: '8px', 
                fontSize: '11px', 
                color: '#9ca3af',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
              }}>
                <span style={{ 
                  display: 'inline-block', 
                  width: '60px', 
                  height: '8px', 
                  background: 'linear-gradient(to right, blue, cyan, lime, yellow, red)',
                  borderRadius: '2px',
                }} />
                <span>Low → High xT</span>
              </div>
            </div>
          )}

          {/* Debug Test Button */}
          {onTest && (
            <>
              <button
                onClick={async () => { try { await onTest(); } catch (e) { console.error(e); } }}
                disabled={testLoading}
                style={{
                  marginTop: '12px',
                  padding: '8px',
                  background: testLoading ? '#6b7280' : '#374151',
                  color: 'white',
                  border: '1px solid #4b5563',
                  borderRadius: '4px',
                  cursor: testLoading ? 'not-allowed' : 'pointer',
                  fontSize: '12px',
                  width: '100%'
                }}
              >
                {testLoading ? 'Testing…' : '🧪 Debug: Test Endpoint'}
              </button>
              {testResult && (
                <div style={{ marginTop: 8, padding: 8, background: '#1e293b', borderRadius: 6, fontSize: 11, maxHeight: '200px', overflowY: 'auto' }}>
                  <div style={{ fontWeight: 700, marginBottom: 6 }}>Debug Output</div>
                  <pre style={{ whiteSpace: 'pre-wrap', margin: 0, fontSize: 10 }}>{JSON.stringify(testResult, null, 2)}</pre>
                </div>
              )}
            </>
          )}

          {/* Debug Pass Test Button */}
          {onTestPass && (
            <>
              <button
                onClick={async () => { try { await onTestPass(); } catch (e) { console.error(e); } }}
                disabled={passTestLoading}
                style={{
                  marginTop: '8px',
                  padding: '8px',
                  background: passTestLoading ? '#6b7280' : '#4c1d95',
                  color: 'white',
                  border: '1px solid #6d28d9',
                  borderRadius: '4px',
                  cursor: passTestLoading ? 'not-allowed' : 'pointer',
                  fontSize: '12px',
                  width: '100%'
                }}
              >
                {passTestLoading ? 'Testing…' : '📤 Debug: Pass Likelihoods'}
              </button>
              {passTestResult && (
                <div style={{ marginTop: 8, padding: 8, background: '#1e293b', borderRadius: 6, fontSize: 11, maxHeight: '300px', overflowY: 'auto' }}>
                  <div style={{ fontWeight: 700, marginBottom: 6 }}>Pass Likelihood Debug (Raw)</div>
                  <pre style={{ whiteSpace: 'pre-wrap', margin: 0, fontSize: 10 }}>{JSON.stringify(passTestResult, null, 2)}</pre>
                </div>
              )}
            </>
          )}
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
