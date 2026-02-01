import { useState } from 'react';
import type { Player, XTResult } from '../types';
import type { Preset } from '../presets';
import type { GenerationStatus } from '../types';
import { COLORS } from '../constants';
import { FORMATION_PRESETS, SITUATION_PRESETS } from '../presets';

type MenuBarProps = {
  players: Player[];
  isOpen: boolean;
  ballCarrier: string | null;
  onOpenChange: (open: boolean) => void;
  onBallCarrierChange: (playerId: string | null) => void;
  onLoadPreset: (preset: Preset) => void;
  onGenerateCustom: (situation: string) => Promise<void>;
  generationStatus: GenerationStatus;
  onCalculateXT: () => Promise<void>;
  xTResult: XTResult | null;
  xTLoading: boolean;
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
}: MenuBarProps) {
  const [customSituation, setCustomSituation] = useState('');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageLoading, setImageLoading] = useState(false);
  const [imageResponse, setImageResponse] = useState<any | null>(null);
  const attackers = players.filter((p) => p.type === 'attacker');

  const handleGenerate = async () => {
    if (!customSituation.trim()) return;
    await onGenerateCustom(customSituation);
  };

  const readFileAsBase64 = (file: File) => new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // data:<mime>;base64,<data>
      const idx = result.indexOf(',');
      const base64 = idx >= 0 ? result.slice(idx + 1) : result;
      resolve(base64);
    };
    reader.onerror = (e) => reject(e);
    reader.readAsDataURL(file);
  });

  const handleUploadImage = async () => {
    if (!imageFile) return;
    setImageLoading(true);
    setImageResponse(null);
    try {
      const base64 = await readFileAsBase64(imageFile);
      const payload = { image: base64 };
      const resp = await fetch('http://localhost:5001/image', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const data = await resp.json();
      setImageResponse(data);
    } catch (err) {
      console.error('Image upload failed', err);
      const msg = err instanceof Error ? err.message : String(err);
      setImageResponse({ error: `Upload failed: ${msg}. Is the backend running on http://localhost:5001 ?` });
    } finally {
      setImageLoading(false);
    }
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
            value={ballCarrier ?? ''}
            onChange={(e) => onBallCarrierChange(e.target.value || null)}
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
        {/* Image uploader */}
        <div style={{ marginBottom: '18px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '8px' }}>Upload Image</h3>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImageFile(e.target.files && e.target.files[0] ? e.target.files[0] : null)}
            style={{ width: '100%', marginBottom: 8 }}
          />
          <button
            onClick={handleUploadImage}
            disabled={!imageFile || imageLoading}
            style={{
              padding: '8px 10px',
              background: imageLoading ? '#6b7280' : '#06b6d4',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: !imageFile || imageLoading ? 'not-allowed' : 'pointer',
              fontSize: '13px',
              width: '100%'
            }}
          >
            {imageLoading ? 'Uploading…' : 'Upload image to backend'}
          </button>

          {imageResponse && (
            <div style={{ marginTop: 8, padding: 8, background: '#061025', borderRadius: 6, fontSize: 12 }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Response</div>
              <pre style={{ whiteSpace: 'pre-wrap', margin: 0, fontSize: 12 }}>{JSON.stringify(imageResponse, null, 2)}</pre>
            </div>
          )}
        </div>

        {/* Custom Situation Generator */}
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

          {generationStatus === 'error' && (
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


        {/* Calculate xT Section */}
        <div style={{ marginBottom: '24px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '12px' }}>Expected Threat (xT)</h3>
          <button
            onClick={onCalculateXT}
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
            {xTLoading ? 'Calculating...' : 'Calculate xT'}
          </button>

          {xTResult && (
            <div style={{
              marginTop: '12px',
              padding: '12px',
              background: xTResult.error ? '#7f1d1d' : '#065f46',
              borderRadius: '4px',
              fontSize: '13px',
            }}>
              {xTResult.error ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ fontSize: '16px' }}>✗</span>
                  <span>{xTResult.error}</span>
                </div>
              ) : (
                <div>
                  <div style={{ fontWeight: 'bold', marginBottom: '6px' }}>xT Value:</div>
                  <div style={{ fontSize: '20px', fontWeight: 'bold' }}>
                    {typeof xTResult === 'number' 
                      ? xTResult.toFixed(4)
                      : typeof xTResult.xT === 'number'
                        ? xTResult.xT.toFixed(4)
                        : JSON.stringify(xTResult, null, 2)}
                  </div>
                </div>
              )}
            </div>
          )}
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
