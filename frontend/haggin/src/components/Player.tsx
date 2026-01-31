import { useState } from 'react';
import type { Player as PlayerType } from '../types';
import { COLORS, PLAYER_RADIUS, PLAYER_RING_OFFSET } from '../constants';

type PlayerProps = {
  player: PlayerType;
  scale: number;
  hasBall: boolean;
  onDragEnd: (id: string, x: number, y: number) => void;
};

export function Player({ player, scale, hasBall, onDragEnd }: PlayerProps) {
  const [isDragging, setIsDragging] = useState(false);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const svg = (e.target as SVGElement).closest('svg');
      if (!svg) return;

      const rect = svg.getBoundingClientRect();
      const x = (moveEvent.clientX - rect.left) / scale;
      const y = (moveEvent.clientY - rect.top) / scale;

      onDragEnd(player.id, x, y);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const outlineColor = player.type === 'attacker'
    ? COLORS.attackerOutline
    : COLORS.defenderOutline;

  // Priority: dragging (black) > ball carrier (yellow) > team color
  const ringColor = isDragging ? '#000' : hasBall ? '#fbbf24' : outlineColor;

  // Extract player number from ID (e.g., 'd1' -> '1', '10' -> '10')
  const playerNumber = player.id.replace(/^d/, '');

  return (
    <g>
      {/* Outline ring (black for dragging, yellow for ball, team color otherwise) */}
      <circle
        cx={player.position.x}
        cy={player.position.y}
        r={PLAYER_RADIUS + PLAYER_RING_OFFSET}
        fill="none"
        stroke={ringColor}
        strokeWidth={0.6}
      />
      {/* Player dot */}
      <circle
        cx={player.position.x}
        cy={player.position.y}
        r={PLAYER_RADIUS}
        fill={COLORS[player.type]}
        style={{ cursor: 'move' }}
        onMouseDown={handleMouseDown}
      />
      {/* Player number */}
      <text
        x={player.position.x}
        y={player.position.y}
        textAnchor="middle"
        dominantBaseline="central"
        fill="white"
        fontSize={1.2}
        fontWeight="bold"
        style={{ pointerEvents: 'none', userSelect: 'none' }}
      >
        {playerNumber}
      </text>
    </g>
  );
}
