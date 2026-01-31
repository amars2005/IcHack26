// Pitch dimensions in metres
export const PITCH_WIDTH = 120;
export const PITCH_HEIGHT = 80;

// Goal position (right side, centered vertically)
export const GOAL_X = 120;
export const GOAL_Y = 40;

// Visual styling
export const COLORS = {
  attacker: '#ef4444',      // Red
  defender: '#3b82f6',      // Blue
  attackerOutline: '#7f1d1d',  // Dark red
  defenderOutline: '#1e3a8a',  // Dark blue
  pitch: '#22c55e',         // Green
  pitchBorder: '#16a34a',   // Dark green
  lines: '#ffffff',         // White
};

// Player visual settings
export const PLAYER_RADIUS = 1.5;
export const PLAYER_RING_OFFSET = 0.3;

// Initial ball carrier
export const INITIAL_BALL_CARRIER = '9';

// 4-3-3 Formation for attackers (home team - left side, attacking right)
// Goalkeeper, 4 defenders, 3 midfielders, 3 forwards
// Halfway line is at x=60
const HOME_FORMATION = [
  { id: '1', position: { x: 12, y: 40 } },   // GK
  { id: '2', position: { x: 28, y: 64 } },   // RB
  { id: '3', position: { x: 28, y: 48 } },   // CB
  { id: '4', position: { x: 28, y: 32 } },   // CB
  { id: '5', position: { x: 28, y: 16 } },   // LB
  { id: '6', position: { x: 46, y: 52 } },   // RCM
  { id: '7', position: { x: 46, y: 40 } },   // CM
  { id: '8', position: { x: 46, y: 28 } },   // LCM
  { id: '9', position: { x: 56, y: 58 } },   // RW
  { id: '10', position: { x: 56, y: 40 } },  // ST
  { id: '11', position: { x: 56, y: 22 } },  // LW
];

// 4-3-3 Formation for defenders (away team - right side, defending right)
const AWAY_FORMATION = [
  { id: 'd1', position: { x: 108, y: 40 } }, // GK
  { id: 'd2', position: { x: 92, y: 16 } },  // LB
  { id: 'd3', position: { x: 92, y: 32 } },  // CB
  { id: 'd4', position: { x: 92, y: 48 } },  // CB
  { id: 'd5', position: { x: 92, y: 64 } },  // RB
  { id: 'd6', position: { x: 74, y: 28 } },  // LCM
  { id: 'd7', position: { x: 74, y: 40 } },  // CM
  { id: 'd8', position: { x: 74, y: 52 } },  // RCM
  { id: 'd9', position: { x: 64, y: 22 } },  // LW
  { id: 'd10', position: { x: 64, y: 40 } }, // ST
  { id: 'd11', position: { x: 64, y: 58 } }, // RW
];

// Initial player positions
export const INITIAL_PLAYERS = [
  ...HOME_FORMATION.map(p => ({ ...p, type: 'attacker' as const })),
  ...AWAY_FORMATION.map(p => ({ ...p, type: 'defender' as const })),
];
