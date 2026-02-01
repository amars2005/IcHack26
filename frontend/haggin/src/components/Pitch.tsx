import { useState } from 'react';
import { Player } from './Player';
import type { Player as PlayerType, HeatmapData } from '../types';
import { PITCH_WIDTH, PITCH_HEIGHT, COLORS, GOAL_Y } from '../constants';

type HoverInfo = {
  x: number;
  y: number;
  value: number;
} | null;

type PitchProps = {
  players: PlayerType[];
  ballCarrier: string;
  onPlayerMove: (id: string, x: number, y: number) => void;
  scale?: number;
  teamColor?: string;
  opponentColor?: string;
  onAssignBall?: (id: string) => void;
  heatmap?: HeatmapData | null;
  showHeatmap?: boolean;
  moves?: Array<{ id: string; type: string; targetId?: string | null }>;
  onExecuteMove?: (move: { id: string; type: string; targetId?: string | null }) => void;
};

// Interpolate color between blue (low) -> yellow (mid) -> red (high)
function getHeatmapColor(value: number, min: number, max: number): string {
  const normalized = Math.max(0, Math.min(1, (value - min) / (max - min || 1)));
  
  // Blue (0) -> Cyan (0.25) -> Green (0.5) -> Yellow (0.75) -> Red (1)
  let r: number, g: number, b: number;
  
  if (normalized < 0.25) {
    const t = normalized / 0.25;
    r = 0; g = Math.round(255 * t); b = 255;
  } else if (normalized < 0.5) {
    const t = (normalized - 0.25) / 0.25;
    r = 0; g = 255; b = Math.round(255 * (1 - t));
  } else if (normalized < 0.75) {
    const t = (normalized - 0.5) / 0.25;
    r = Math.round(255 * t); g = 255; b = 0;
  } else {
    const t = (normalized - 0.75) / 0.25;
    r = 255; g = Math.round(255 * (1 - t)); b = 0;
  }
  
  return `rgb(${r}, ${g}, ${b})`;
}

export function Pitch({ players, ballCarrier, onPlayerMove, scale = 1, heatmap, showHeatmap = true, teamColor, opponentColor, onAssignBall, moves = [], onExecuteMove }: PitchProps) {
  const [hoverInfo, setHoverInfo] = useState<HoverInfo>(null);
  const goalWidth = 20;
  const penaltyBoxWidth = 18;
  const penaltyBoxHeight = 44;
  // Small goal-area box (approx. 6-yard box)
  const smallBoxWidth = 6;
  const smallBoxHeight = 20;
  // Penalty arc geometry: radius and computed intersection with penalty box edge
  const penaltySpotLeftX = 12;
  const penaltyBoxLeftEdge = penaltyBoxWidth; // x coordinate where penalty area ends on left
  const penaltyBoxRightEdge = PITCH_WIDTH - penaltyBoxWidth;
  const penaltyArcR = 10;
  const penaltyArcDx = penaltyBoxLeftEdge - penaltySpotLeftX; // should be positive (6)
  const penaltyArcDy = Math.sqrt(Math.max(0, penaltyArcR * penaltyArcR - penaltyArcDx * penaltyArcDx));
  const leftArcStartX = penaltyBoxLeftEdge;
  const leftArcStartY = PITCH_HEIGHT / 2 - penaltyArcDy;
  const leftArcEndX = penaltyBoxLeftEdge;
  const leftArcEndY = PITCH_HEIGHT / 2 + penaltyArcDy;
  const rightArcStartX = penaltyBoxRightEdge;
  const rightArcStartY = PITCH_HEIGHT / 2 - penaltyArcDy;
  const rightArcEndX = penaltyBoxRightEdge;
  const rightArcEndY = PITCH_HEIGHT / 2 + penaltyArcDy;

  return (
  <div style={{ position: 'relative', display: 'inline-block' }}>
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

      {/* SVG filter for smooth heatmap */}
      <defs>
        <filter id="heatmapBlur" x="-10%" y="-10%" width="120%" height="120%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="1.5" />
        </filter>
      </defs>

      {/* Heatmap overlay */}
      {showHeatmap && heatmap && heatmap.heatmap.length > 0 && (() => {
        const { heatmap: grid, x_coords, y_coords } = heatmap;
        const allValues = grid.flat();
        const minVal = Math.min(...allValues);
        const maxVal = Math.max(...allValues);
        
        // Calculate cell dimensions with slight overlap for smoother appearance
        const cellWidth = x_coords.length > 1 
          ? (x_coords[x_coords.length - 1] - x_coords[0]) / (x_coords.length - 1) * 1.15
          : PITCH_WIDTH / x_coords.length;
        const cellHeight = y_coords.length > 1
          ? (y_coords[y_coords.length - 1] - y_coords[0]) / (y_coords.length - 1) * 1.15
          : PITCH_HEIGHT / y_coords.length;

        return (
          <>
            {/* Visual heatmap layer with blur */}
            <g opacity={0.55} filter="url(#heatmapBlur)">
              {grid.map((row, rowIdx) =>
                row.map((value, colIdx) => {
                  // Backend uses y=0 at bottom, SVG uses y=0 at top
                  const x = x_coords[colIdx] - cellWidth / 2;
                  const y = PITCH_HEIGHT - y_coords[rowIdx] - cellHeight / 2;
                  
                  return (
                    <rect
                      key={`heatmap-${rowIdx}-${colIdx}`}
                      x={Math.max(0, x)}
                      y={Math.max(0, y)}
                      width={cellWidth}
                      height={cellHeight}
                      fill={getHeatmapColor(value, minVal, maxVal)}
                    />
                  );
                })
              )}
            </g>
            {/* Invisible interaction layer for hover events */}
            <g opacity={0}>
              {grid.map((row, rowIdx) =>
                row.map((value, colIdx) => {
                  const x = x_coords[colIdx] - cellWidth / 2;
                  const y = PITCH_HEIGHT - y_coords[rowIdx] - cellHeight / 2;
                  const cellX = Math.max(0, x);
                  const cellY = Math.max(0, y);
                  
                  return (
                    <rect
                      key={`heatmap-hover-${rowIdx}-${colIdx}`}
                      x={cellX}
                      y={cellY}
                      width={cellWidth}
                      height={cellHeight}
                      fill="transparent"
                      style={{ cursor: 'crosshair' }}
                      onMouseEnter={() => setHoverInfo({ 
                        x: (cellX + cellWidth / 2) * scale, 
                        y: cellY * scale, 
                        value 
                      })}
                      onMouseLeave={() => setHoverInfo(null)}
                    />
                  );
                })
              )}
            </g>
          </>
        );
      })()}

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

      {/* Centre spot for kick-off */}
      <circle cx={PITCH_WIDTH / 2} cy={PITCH_HEIGHT / 2} r={0.6} fill={COLORS.lines} />

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

      {/* Small goal-area boxes (6-yard boxes) */}
      <rect
        x={0}
        y={PITCH_HEIGHT / 2 - smallBoxHeight / 2}
        width={smallBoxWidth}
        height={smallBoxHeight}
        fill="none"
        stroke={COLORS.lines}
        strokeWidth={0.5}
        rx={0.3}
      />
      <rect
        x={PITCH_WIDTH - smallBoxWidth}
        y={PITCH_HEIGHT / 2 - smallBoxHeight / 2}
        width={smallBoxWidth}
        height={smallBoxHeight}
        fill="none"
        stroke={COLORS.lines}
        strokeWidth={0.5}
        rx={0.3}
      />

      {/* Penalty spots */}
      <circle cx={12} cy={PITCH_HEIGHT / 2} r={0.6} fill={COLORS.lines} />
      <circle cx={PITCH_WIDTH - 12} cy={PITCH_HEIGHT / 2} r={0.6} fill={COLORS.lines} />

      {/* Penalty arcs (the 'D') - drawn only outside the penalty box edges */}
      <path
        d={`M ${leftArcStartX} ${leftArcStartY} A ${penaltyArcR} ${penaltyArcR} 0 0 1 ${leftArcEndX} ${leftArcEndY}`}
        stroke={COLORS.lines}
        strokeWidth={0.5}
        fill="none"
      />
      <path
        d={`M ${rightArcStartX} ${rightArcStartY} A ${penaltyArcR} ${penaltyArcR} 0 0 0 ${rightArcEndX} ${rightArcEndY}`}
        stroke={COLORS.lines}
        strokeWidth={0.5}
        fill="none"
      />

      {/* Optimal action arrow: rendered after players to overlay them (moved below) */}

      {/* Players */}
      {players.map((player) => (
        <Player
          key={player.id}
          player={player}
          scale={scale}
          hasBall={player.id === ballCarrier}
          onDragEnd={onPlayerMove}
          teamColor={teamColor}
          opponentColor={opponentColor}
          onRightClick={onAssignBall}
        />
      ))}

      {/* --- Optimal action arrow (render last so it overlays players) --- */}
      {(() => {
        try {
          if (!moves || moves.length === 0) return null;
          const best = moves[0];
          if (!best) return null;

          const origin = players.find(p => p.id === ballCarrier);
          if (!origin) return null;

          let tx: number | null = null;
          let ty: number | null = null;

          if (best.type === 'pass' && best.targetId) {
            const tgt = players.find(p => p.id === best.targetId);
            if (tgt) { tx = tgt.position.x; ty = tgt.position.y; }
          } else if (best.type === 'shoot') {
            tx = PITCH_WIDTH; ty = GOAL_Y;
          } else if (best.type === 'dribble') {
            tx = Math.min(PITCH_WIDTH, origin.position.x + 12);
            ty = origin.position.y;
          }

          if (tx == null || ty == null) return null;

          const x1 = origin.position.x;
          const y1 = origin.position.y;
          const x2 = tx;
          const y2 = ty;

          const dx = x2 - x1;
          const dy = y2 - y1;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;

          const nx = dx / dist;
          const ny = dy / dist;
          const px = -ny;
          const py = nx;
          const curve = Math.min(18, dist * 0.18);
          const cx = (x1 + x2) / 2 + px * curve;
          const cy = (y1 + y2) / 2 + py * curve;

          const pathD = `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;

          const arrowColor = '#fbbf24';

          const txv = 2 * (x2 - cx);
          const tyv = 2 * (y2 - cy);
          const tlen = Math.sqrt(txv * txv + tyv * tyv) || 1;
          const ux = txv / tlen;
          const uy = tyv / tlen;
          const headLen = Math.max(4, Math.min(10, dist * 0.04));
          const baseX = x2 - ux * headLen;
          const baseY = y2 - uy * headLen;
          const sideX = -uy;
          const sideY = ux;
          const halfW = headLen * 0.38;
          const p1x = baseX + sideX * halfW;
          const p1y = baseY + sideY * halfW;
          const p2x = baseX - sideX * halfW;
          const p2y = baseY - sideY * halfW;

          return (
            <g onClick={() => { try { if (onExecuteMove) onExecuteMove(best); } catch (e) { console.error(e); } }} style={{ cursor: 'pointer' }}>
              <path d={pathD} stroke={arrowColor} strokeWidth={1.2} strokeOpacity={0.22} strokeLinecap="round" strokeLinejoin="round" fill="none" />
              <path d={pathD} stroke={arrowColor} strokeWidth={0.9} strokeLinecap="round" strokeLinejoin="round" fill="none" />
              <path d={pathD} stroke="#fff" strokeWidth={0.35} strokeOpacity={0.18} strokeLinecap="round" strokeLinejoin="round" fill="none" />
              <polygon points={`${x2},${y2} ${p1x},${p1y} ${p2x},${p2y}`} fill={arrowColor} opacity={1} />
            </g>
          );
        } catch (e) {
          console.error('Move arrow render error', e);
          return null;
        }
      })()}
    </svg>
    
    {/* xT Value Tooltip */}
    {showHeatmap && hoverInfo && (
      <div
        style={{
          position: 'absolute',
          left: `${hoverInfo.x + 15}px`,
          top: `${hoverInfo.y - 10}px`,
          background: 'rgba(15, 23, 42, 0.95)',
          color: 'white',
          padding: '8px 12px',
          borderRadius: '6px',
          fontSize: '12px',
          pointerEvents: 'none',
          zIndex: 100,
          maxWidth: '200px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          border: '1px solid rgba(255,255,255,0.1)',
        }}
      >
        <div style={{ fontWeight: 'bold', marginBottom: '4px', color: '#f59e0b' }}>
          xT: {(hoverInfo.value * 100).toFixed(2)}%
        </div>
        <div style={{ color: '#9ca3af', lineHeight: '1.4' }}>
          Chance of ball starting here ending up in a goal eventually
        </div>
      </div>
    )}
  </div>
  );
}
