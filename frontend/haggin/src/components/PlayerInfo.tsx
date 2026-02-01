import type { Position } from '../types';
import { PITCH_WIDTH, PITCH_HEIGHT } from '../constants';

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
        <div style={{ marginTop: '8px', color: '#cbd5e1' }}>
          <div style={{ fontSize: 12, color: '#9ca3af' }}>Position</div>
          <div style={{ fontSize: 14, fontWeight: 700 }}>
            {(() => {
              const x = playerPosition.x;
              const y = playerPosition.y;
              if (x <= 12) return 'Goalkeeper';

              const thirdX = PITCH_WIDTH / 3;
              const lateral = (() => {
                const t = PITCH_HEIGHT / 3;
                if (y < t) return 'Left';
                if (y < t * 2) return 'Centre';
                return 'Right';
              })();

              const depth = x < thirdX ? 'Defensive' : x < thirdX * 2 ? 'Midfield' : 'Attacking';

              if (depth === 'Defensive') {
                if (lateral === 'Left') return 'Left Back';
                if (lateral === 'Centre') return 'Center Back';
                return 'Right Back';
              }
              if (depth === 'Midfield') {
                if (lateral === 'Left') return 'Left Mid';
                if (lateral === 'Centre') return 'Centre Mid';
                return 'Right Mid';
              }
              if (lateral === 'Left') return 'Left Wing';
              if (lateral === 'Centre') return 'Striker';
              return 'Right Wing';
            })()}
          </div>
        </div>
      )}

      <div style={{ marginTop: '14px', display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'stretch' }}>
        {[
          { key: 'xG', label: 'xG' },
          { key: 'xT', label: 'xT' },
          { key: 'Shots', label: 'Shots' },
          { key: 'Passes', label: 'Passes' },
          { key: 'Progressive', label: 'Prog' },
        ].map(({ key, label }) => {
          const raw = metrics && metrics[key] != null ? metrics[key] : null;
          const display = raw == null ? '-' : (typeof raw === 'number' ? (raw >= 1 || key === 'Shots' || key === 'Passes' ? String(Math.round(raw)) : Number(raw).toFixed(2)) : String(raw));
          return (
            <div key={key} style={{ padding: '8px 10px', background: '#061021', borderRadius: '8px', minWidth: 78, display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
              <div style={{ fontSize: '11px', color: '#9ca3af' }}>{label}</div>
              <div style={{ fontSize: '15px', fontWeight: 700, color: '#e6eef8' }}>{display}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
import type { Position } from '../types';
import { PITCH_WIDTH, PITCH_HEIGHT } from '../constants';

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
        <div style={{ marginTop: '8px', color: '#cbd5e1' }}>
          <div style={{ fontSize: 12, color: '#9ca3af' }}>Position</div>
          <div style={{ fontSize: 14, fontWeight: 700 }}>
            {(() => {
              const x = playerPosition.x;
              const y = playerPosition.y;
              // Special case: goalkeeper area (very close to left goal)
              if (x <= 12) return 'Goalkeeper';

              const thirdX = PITCH_WIDTH / 3; // defensive / middle / attacking
              const lateral = (() => {
                const t = PITCH_HEIGHT / 3;
                if (y < t) return 'Left';
                if (y < t * 2) return 'Centre';
                return 'Right';
              })();

              const depth = x < thirdX ? 'Defensive' : x < thirdX * 2 ? 'Midfield' : 'Attacking';

              // Map to common football role names
              if (depth === 'Defensive') {
                if (lateral === 'Left') return 'Left Back';
                if (lateral === 'Centre') return 'Center Back';
                return 'Right Back';
              }
              if (depth === 'Midfield') {
                if (lateral === 'Left') return 'Left Mid';
                if (lateral === 'Centre') return 'Centre Mid';
                return 'Right Mid';
              }
              // Attacking
              if (lateral === 'Left') return 'Left Wing';
              if (lateral === 'Centre') return 'Striker';
              return 'Right Wing';
            })()}
          </div>
        </div>
      )}

      <div style={{ marginTop: '14px', display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'stretch' }}>
        {/* Compact, fixed set of important stats */}
        {[
          { key: 'xG', label: 'xG' },
          { key: 'xT', label: 'xT' },
          { key: 'Shots', label: 'Shots' },
          { key: 'Passes', label: 'Passes' },
          { key: 'Progressive', label: 'Prog' },
        ].map(({ key, label }) => {
          const raw = metrics && metrics[key] != null ? metrics[key] : null;
          const display = raw == null ? '-' : (typeof raw === 'number' ? (raw >= 1 || key === 'Shots' || key === 'Passes' ? String(Math.round(raw)) : Number(raw).toFixed(2)) : String(raw));
          return (
            <div key={key} style={{ padding: '8px 10px', background: '#061021', borderRadius: '8px', minWidth: 78, display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
              <div style={{ fontSize: '11px', color: '#9ca3af' }}>{label}</div>
              <div style={{ fontSize: '15px', fontWeight: 700, color: '#e6eef8' }}>{display}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
