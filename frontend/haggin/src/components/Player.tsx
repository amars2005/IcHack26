import { useState } from 'react';
import type React from 'react';
import { motion } from 'framer-motion';
import type { Player as PlayerType } from '../types';
import { COLORS, PLAYER_RADIUS, PLAYER_RING_OFFSET } from '../constants';

type PlayerProps = {
  player: PlayerType;
  scale: number;
  hasBall: boolean;
  onDragEnd: (id: string, x: number, y: number) => void;
  onRightClick?: (id: string) => void;
  teamColor?: string;
  opponentColor?: string;
};

export function Player({ player, scale, hasBall, onDragEnd, onRightClick, teamColor, opponentColor }: PlayerProps) {
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

      // Update position immediately during drag
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

  // Priority: dragging (black) > ball carrier (yellow) > team/opponent color
  const ringColor = isDragging
    ? '#000'
    : hasBall
      ? '#fbbf24'
      : (player.type === 'attacker' ? (teamColor || COLORS.attacker) : (opponentColor || COLORS.defender));

  // Extract player number from ID (e.g., 'd1' -> '1', '10' -> '10')
  const playerNumber = player.id.replace(/^d/, '');

  const displayX = player.position.x;
  const displayY = player.position.y;

  // Animation configuration - smooth spring with exponential decay
  // Disable animation during drag for immediate response
  const animationConfig = isDragging
    ? { duration: 0 } // Immediate during drag
    : {
        type: "spring" as const,
        damping: 20,
        stiffness: 100,
        mass: 1,
      };

  return (
    <motion.g
      animate={{
        x: displayX,
        y: displayY,
      }}
      transition={animationConfig}
      onContextMenu={(e: React.MouseEvent) => {
        e.preventDefault();
        onRightClick?.(player.id);
      }}
    >
      {/* Outline ring (black for dragging, yellow for ball, team color otherwise) */}
      <circle
        cx={0}
        cy={0}
        r={PLAYER_RADIUS + PLAYER_RING_OFFSET}
        fill="none"
        stroke={ringColor}
        strokeWidth={0.6}
      />
      {/* Player dot */}
      <circle
        cx={0}
        cy={0}
        r={PLAYER_RADIUS}
        fill={player.type === 'attacker' ? (teamColor || COLORS.attacker) : (opponentColor || COLORS.defender)}
        style={{ cursor: 'move' }}
        onMouseDown={handleMouseDown}
      />
      {/* Player number - centered using SVG styling */}
      <text
        x={0}
        y={0}
        textAnchor="middle"
        dominantBaseline="central"
        fill="white"
        fontSize={1.56}
        fontWeight="bold"
        style={{ pointerEvents: 'none', userSelect: 'none' }}
      >
        {playerNumber}
      </text>
    </motion.g>
  );
}
