import type { Position } from '../types';
import { PITCH_WIDTH, PITCH_HEIGHT } from '../constants';

type PlayerInfoProps = {
  playerId: string | null;
  playerPosition?: Position | null;
};

function roleFromPosition(pos?: Position | null) {
  if (!pos) return '—';
  const x = pos.x;
  const y = pos.y;
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
}

export function PlayerInfo({ playerId, playerPosition }: PlayerInfoProps) {
  return (
    <div style={{
      width: '100%',
      marginTop: 0,
      padding: '8px 12px',
      background: '#0b1220',
      color: 'white',
      borderRadius: '10px',
      boxShadow: '0 6px 30px rgba(0,0,0,0.6)',
      boxSizing: 'border-box'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
          <div style={{ fontSize: 12, color: '#9ca3af' }}>On the ball</div>
          <div style={{ fontSize: 20, fontWeight: 700 }}>{playerId ? `#${playerId}` : '—'}</div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
          <div style={{ fontSize: 12, color: '#9ca3af' }}>Position</div>
          <div style={{ fontSize: 16, fontWeight: 700 }}>{roleFromPosition(playerPosition)}</div>
        </div>
      </div>
    </div>
  );
}
