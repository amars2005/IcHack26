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
          <h3 style={{ fontSize: '16px', marginBottom: '8px' }}>Decision Engine</h3>
          <p style={{ fontSize: '12px', color: '#6b7280', marginTop: 0, marginBottom: '12px' }}>
            AI-powered analysis of optimal actions
          </p>
          <button
            onClick={async () => { try { await onCalculateXT(); } catch (e) { console.error(e); } }}
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

          {xTResult && !xTResult.error && (
            <div style={{ marginTop: '16px', background: 'linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%)', borderRadius: '12px', overflow: 'hidden', border: '1px solid #312e81' }}>
              
              {/* Current State Header */}
              {xTResult.current_xT !== undefined && xTResult.current_xT !== null && (
                <div style={{
                  padding: '12px 16px',
                  background: 'rgba(99, 102, 241, 0.1)',
                  borderBottom: '1px solid #312e81',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <span style={{ fontSize: '12px', color: '#a5b4fc' }}>Current Position Value</span>
                  <span style={{ fontSize: '16px', fontWeight: 700, color: '#818cf8' }}>
                    {((xTResult.current_xT as number) * 100).toFixed(2)}%
                  </span>
                </div>
              )}

              {/* Shoot action */}
              {xTResult.shoot && (
                <div style={{ 
                  padding: '14px 16px', 
                  borderBottom: '1px solid #312e81',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  background: xTResult.shoot.xG > 0.1 ? 'rgba(34, 197, 94, 0.1)' : 'transparent'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ 
                      width: '36px', 
                      height: '36px', 
                      borderRadius: '8px', 
                      background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '18px'
                    }}></div>
                    <div>
                      <div style={{ fontWeight: 600, color: '#fbbf24', fontSize: '14px' }}>Shoot</div>
                      <div style={{ fontSize: '10px', color: '#6b7280' }}>Expected Goal</div>
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ 
                      fontSize: '20px', 
                      fontWeight: 700, 
                      color: xTResult.shoot.xG > 0.15 ? '#22c55e' : xTResult.shoot.xG > 0.05 ? '#fbbf24' : '#ef4444'
                    }}>
                      {(xTResult.shoot.xG * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              )}

              {/* Carry action - only show if implemented */}
              {xTResult.carry && xTResult.carry.xT !== null && (
                <div style={{ 
                  padding: '14px 16px', 
                  borderBottom: '1px solid #312e81',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <div style={{ 
                      width: '36px', 
                      height: '36px', 
                      borderRadius: '8px', 
                      background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '18px'
                    }}></div>
                    <div>
                      <div style={{ fontWeight: 600, color: '#60a5fa', fontSize: '14px' }}>Carry</div>
                      <div style={{ fontSize: '10px', color: '#6b7280' }}>Dribble Forward</div>
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '20px', fontWeight: 700, color: '#60a5fa' }}>
                      {typeof xTResult.carry.xT === 'number' ? (xTResult.carry.xT * 100).toFixed(1) + '%' : '—'}
                    </div>
                  </div>
                </div>
              )}

              {/* Pass actions - sorted by score */}
              {(() => {
                const passActions = Object.entries(xTResult)
                  .filter(([key]) => key.startsWith('pass_to_'))
                  .map(([key, value]: [string, any]) => ({
                    playerId: key.replace('pass_to_', ''),
                    xT: value.xT,
                    probability: value['P(success)'],
                    reward: value.reward,
                    risk: value.risk,
                    score: value.score,
                    opponentXT: value.opponent_xT
                  }))
                  .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));

                if (passActions.length === 0) return null;

                const bestPass = passActions[0];
                const otherPasses = passActions.slice(1);

                return (
                  <div style={{ padding: '14px 16px' }}>
                    <div style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '10px', 
                      marginBottom: '12px',
                    }}>
                      <div style={{ 
                        width: '36px', 
                        height: '36px', 
                        borderRadius: '8px', 
                        background: 'linear-gradient(135deg, #a855f7 0%, #7c3aed 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '18px'
                      }}></div>
                      <div>
                        <div style={{ fontWeight: 600, color: '#c084fc', fontSize: '14px' }}>Pass Options</div>
                        <div style={{ fontSize: '10px', color: '#6b7280' }}>Ranked by expected value</div>
                      </div>
                    </div>

                    {/* Best Pass Highlighted */}
                    {bestPass && (
                      <div style={{
                        background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(16, 185, 129, 0.1) 100%)',
                        borderRadius: '8px',
                        padding: '12px',
                        marginBottom: '10px',
                        border: '1px solid rgba(34, 197, 94, 0.3)'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{ fontSize: '12px', background: '#22c55e', color: '#000', padding: '2px 6px', borderRadius: '4px', fontWeight: 700 }}>BEST</span>
                            <span style={{ fontWeight: 600, fontSize: '15px' }}>Pass to #{bestPass.playerId}</span>
                          </div>
                          <div style={{ 
                            fontSize: '18px', 
                            fontWeight: 700, 
                            color: bestPass.score > 0 ? '#22c55e' : '#ef4444'
                          }}>
                            {bestPass.score > 0 ? '+' : ''}{(bestPass.score * 100).toFixed(2)}
                          </div>
                        </div>
                        <div style={{ display: 'flex', gap: '16px', fontSize: '11px' }}>
                          <div>
                            <span style={{ color: '#6b7280' }}>Success: </span>
                            <span style={{ color: '#22c55e', fontWeight: 600 }}>{(bestPass.probability * 100).toFixed(0)}%</span>
                          </div>
                          <div>
                            <span style={{ color: '#6b7280' }}>Gain: </span>
                            <span style={{ color: bestPass.reward > 0 ? '#22c55e' : '#ef4444', fontWeight: 600 }}>
                              {bestPass.reward > 0 ? '+' : ''}{(bestPass.reward * 100).toFixed(2)}
                            </span>
                          </div>
                          <div>
                            <span style={{ color: '#6b7280' }}>Risk: </span>
                            <span style={{ color: '#f87171', fontWeight: 600 }}>{(bestPass.risk * 100).toFixed(2)}</span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Other Passes Table */}
                    {otherPasses.length > 0 && (
                      <div style={{ fontSize: '11px' }}>
                        <div style={{ 
                          display: 'grid', 
                          gridTemplateColumns: '50px 1fr 55px 55px 60px', 
                          gap: '4px 6px',
                          padding: '6px 8px',
                          background: 'rgba(0,0,0,0.2)',
                          borderRadius: '6px 6px 0 0',
                          fontWeight: 600,
                          color: '#6b7280',
                          fontSize: '9px',
                          textTransform: 'uppercase',
                          letterSpacing: '0.5px'
                        }}>
                          <div>Target</div>
                          <div>Success</div>
                          <div style={{ textAlign: 'right' }}>Gain</div>
                          <div style={{ textAlign: 'right' }}>Risk</div>
                          <div style={{ textAlign: 'right' }}>Score</div>
                        </div>
                        {otherPasses.map(({ playerId, probability, reward, risk, score }, idx) => (
                          <div 
                            key={playerId}
                            style={{ 
                              display: 'grid', 
                              gridTemplateColumns: '50px 1fr 55px 55px 60px', 
                              gap: '4px 6px',
                              padding: '8px',
                              background: idx % 2 === 0 ? 'rgba(0,0,0,0.1)' : 'transparent',
                              borderRadius: idx === otherPasses.length - 1 ? '0 0 6px 6px' : '0',
                              alignItems: 'center'
                            }}
                          >
                            <div style={{ fontWeight: 500 }}>#{playerId}</div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                              <div style={{ 
                                flex: 1, 
                                height: '4px', 
                                background: '#374151', 
                                borderRadius: '2px',
                                overflow: 'hidden'
                              }}>
                                <div style={{ 
                                  width: `${(probability ?? 0) * 100}%`, 
                                  height: '100%', 
                                  background: probability >= 0.8 ? '#22c55e' : probability >= 0.5 ? '#f59e0b' : '#ef4444',
                                  borderRadius: '2px'
                                }} />
                              </div>
                              <span style={{ 
                                fontSize: '10px', 
                                color: probability >= 0.8 ? '#22c55e' : probability >= 0.5 ? '#f59e0b' : '#ef4444',
                                minWidth: '30px',
                                textAlign: 'right',
                                fontWeight: 500
                              }}>
                                {(probability * 100).toFixed(0)}%
                              </span>
                            </div>
                            <div style={{ 
                              textAlign: 'right', 
                              color: reward > 0 ? '#4ade80' : reward < 0 ? '#f87171' : '#6b7280',
                              fontWeight: 500
                            }}>
                              {reward > 0 ? '+' : ''}{(reward * 100).toFixed(1)}
                            </div>
                            <div style={{ textAlign: 'right', color: '#f87171', fontWeight: 500 }}>
                              {(risk * 100).toFixed(1)}
                            </div>
                            <div style={{ 
                              textAlign: 'right', 
                              fontWeight: 600,
                              color: score > 0 ? '#22c55e' : score < -0.01 ? '#ef4444' : '#fbbf24'
                            }}>
                              {score > 0 ? '+' : ''}{(score * 100).toFixed(2)}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Legend */}
                    <div style={{ 
                      marginTop: '12px', 
                      padding: '10px',
                      background: 'rgba(0,0,0,0.2)',
                      borderRadius: '6px',
                      fontSize: '9px',
                      color: '#6b7280',
                      lineHeight: 1.6
                    }}>
                      <div><strong style={{ color: '#9ca3af' }}>Score</strong> = Expected value of the pass action</div>
                      <div><strong style={{ color: '#4ade80' }}>Gain</strong> = Threat increase if successful</div>
                      <div><strong style={{ color: '#f87171' }}>Risk</strong> = Threat loss if intercepted</div>
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
