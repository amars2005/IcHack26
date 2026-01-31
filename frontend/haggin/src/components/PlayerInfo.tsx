import type { Position } from '../types';

type PlayerInfoProps = {
  playerId: string | null;
  playerPosition?: Position | null;
  metrics?: Record<string, number> | null;
  onClose?: () => void;
};

export function PlayerInfo({ playerId, playerPosition, metrics, onClose }: PlayerInfoProps) {
  return (
    <div style={{
      width: '100%',
      marginTop: '18px',
      padding: '14px',
      background: '#0b1220',
      color: 'white',
      borderRadius: '10px',
      boxShadow: '0 6px 30px rgba(0,0,0,0.6)'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <div style={{ fontSize: '14px', color: '#9ca3af' }}>On the ball</div>
          <div style={{ fontSize: '18px', fontWeight: 600 }}>{playerId ? `Player #${playerId}` : 'No selection'}</div>
        </div>
        <div>
          {onClose && (
            <button onClick={onClose} style={{ background: 'transparent', color: '#9ca3af', border: 'none', cursor: 'pointer' }}>Close</button>
          )}
        </div>
      </div>

      {playerPosition && (
        <div style={{ marginTop: '8px', color: '#cbd5e1' }}>Position: x={playerPosition.x.toFixed(1)}, y={playerPosition.y.toFixed(1)}</div>
      )}

      <div style={{ marginTop: '14px', display: 'flex', gap: '14px', flexWrap: 'wrap' }}>
        {metrics && Object.keys(metrics).length > 0 ? (
          Object.entries(metrics).map(([k, v]) => (
            <div key={k} style={{ padding: '10px 14px', background: '#061021', borderRadius: '8px', minWidth: 84 }}>
              <div style={{ fontSize: '12px', color: '#9ca3af' }}>{k}</div>
              <div style={{ fontSize: '16px', fontWeight: 700 }}>{Number(v).toFixed(3)}</div>
            </div>
          ))
        ) : (
          <div style={{ color: '#94a3b8' }}>No metrics available. Click a player to request metrics.</div>
        )}
      </div>
    </div>
  );
}
