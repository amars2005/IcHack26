import { Player } from './Player';
import type { Player as PlayerType } from '../types';
import { PITCH_WIDTH, PITCH_HEIGHT, COLORS, GOAL_Y } from '../constants';

type PitchProps = {
  players: PlayerType[];
  ballCarrier: string;
  onPlayerMove: (id: string, x: number, y: number) => void;
  scale?: number;
};

export function Pitch({ players, ballCarrier, onPlayerMove, scale = 1 }: PitchProps) {
  const goalWidth = 20;
  const penaltyBoxWidth = 18;
  const penaltyBoxHeight = 44;

  return (
    <svg
      width={PITCH_WIDTH * scale}
      height={PITCH_HEIGHT * scale}
      viewBox={`0 0 ${PITCH_WIDTH} ${PITCH_HEIGHT}`}
      style={{
        borderRadius: '4px',
      }}
    >
      {/* Pitch background with border */}
      <rect
        x={0}
        y={0}
        width={PITCH_WIDTH}
        height={PITCH_HEIGHT}
        fill={COLORS.pitch}
        stroke={COLORS.pitchBorder}
        strokeWidth={0.8}
      />

      {/* Corner arcs - all four corners (drawn after background, before other lines) */}
      {/* Bottom-left corner */}
      <path
        d={`M 1.5 0 A 1.5 1.5 0 0 1 0 1.5`}
        stroke={COLORS.lines}
        strokeWidth={0.5}
        fill="none"
      />
      {/* Top-left corner */}
      <path
        d={`M 0 ${PITCH_HEIGHT - 1.5} A 1.5 1.5 0 0 1 1.5 ${PITCH_HEIGHT}`}
        stroke={COLORS.lines}
        strokeWidth={0.5}
        fill="none"
      />
      {/* Bottom-right corner */}
      <path
        d={`M ${PITCH_WIDTH - 1.5} 0 A 1.5 1.5 0 0 0 ${PITCH_WIDTH} 1.5`}
        stroke={COLORS.lines}
        strokeWidth={0.5}
        fill="none"
      />
      {/* Top-right corner */}
      <path
        d={`M ${PITCH_WIDTH} ${PITCH_HEIGHT - 1.5} A 1.5 1.5 0 0 0 ${PITCH_WIDTH - 1.5} ${PITCH_HEIGHT}`}
        stroke={COLORS.lines}
        strokeWidth={0.5}
        fill="none"
      />

      {/* Center line - slightly curved */}
      <path
        d={`M ${PITCH_WIDTH / 2} 0 Q ${PITCH_WIDTH / 2 + 0.3} ${PITCH_HEIGHT / 2} ${PITCH_WIDTH / 2} ${PITCH_HEIGHT}`}
        stroke={COLORS.lines}
        strokeWidth={0.5}
        fill="none"
      />

      {/* Center circle - slightly imperfect */}
      <ellipse
        cx={PITCH_WIDTH / 2}
        cy={PITCH_HEIGHT / 2}
        rx={10}
        ry={10.2}
        fill="none"
        stroke={COLORS.lines}
        strokeWidth={0.5}
      />

      {/* Left goal line emphasis */}
      <line
        x1={0}
        y1={GOAL_Y - goalWidth / 2}
        x2={0}
        y2={GOAL_Y + goalWidth / 2}
        stroke={COLORS.lines}
        strokeWidth={1}
      />

      {/* Right goal line emphasis */}
      <line
        x1={PITCH_WIDTH}
        y1={GOAL_Y - goalWidth / 2}
        x2={PITCH_WIDTH}
        y2={GOAL_Y + goalWidth / 2}
        stroke={COLORS.lines}
        strokeWidth={1}
      />

      {/* Left penalty area - slightly bevelled */}
      <path
        d={`M 0 ${PITCH_HEIGHT / 2 - penaltyBoxHeight / 2} Q 0.3 ${PITCH_HEIGHT / 2 - penaltyBoxHeight / 2 + 0.2} ${penaltyBoxWidth} ${PITCH_HEIGHT / 2 - penaltyBoxHeight / 2} L ${penaltyBoxWidth} ${PITCH_HEIGHT / 2 + penaltyBoxHeight / 2} Q 0.3 ${PITCH_HEIGHT / 2 + penaltyBoxHeight / 2 - 0.2} 0 ${PITCH_HEIGHT / 2 + penaltyBoxHeight / 2}`}
        stroke={COLORS.lines}
        strokeWidth={0.5}
        fill="none"
      />

      {/* Right penalty area - slightly bevelled */}
      <path
        d={`M ${PITCH_WIDTH} ${PITCH_HEIGHT / 2 - penaltyBoxHeight / 2} Q ${PITCH_WIDTH - 0.3} ${PITCH_HEIGHT / 2 - penaltyBoxHeight / 2 + 0.2} ${PITCH_WIDTH - penaltyBoxWidth} ${PITCH_HEIGHT / 2 - penaltyBoxHeight / 2} L ${PITCH_WIDTH - penaltyBoxWidth} ${PITCH_HEIGHT / 2 + penaltyBoxHeight / 2} Q ${PITCH_WIDTH - 0.3} ${PITCH_HEIGHT / 2 + penaltyBoxHeight / 2 - 0.2} ${PITCH_WIDTH} ${PITCH_HEIGHT / 2 + penaltyBoxHeight / 2}`}
        stroke={COLORS.lines}
        strokeWidth={0.5}
        fill="none"
      />

      {/* Players */}
      {players.map((player) => (
        <Player
          key={player.id}
          player={player}
          scale={scale}
          hasBall={player.id === ballCarrier}
          onDragEnd={onPlayerMove}
        />
      ))}
    </svg>
  );
}
