import type { Player } from './types';

export type Preset = {
  name: string;
  players: Player[];
  ballCarrier: string;
};

// Starting Formations

export const FORMATION_4_3_3: Preset = {
  name: '4-3-3',
  ballCarrier: '9',
  players: [
    // Home team (attacking right)
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },   // GK
    { id: '2', type: 'attacker', position: { x: 28, y: 64 } },   // RB
    { id: '3', type: 'attacker', position: { x: 28, y: 48 } },   // CB
    { id: '4', type: 'attacker', position: { x: 28, y: 32 } },   // CB
    { id: '5', type: 'attacker', position: { x: 28, y: 16 } },   // LB
    { id: '6', type: 'attacker', position: { x: 46, y: 52 } },   // RCM
    { id: '7', type: 'attacker', position: { x: 46, y: 40 } },   // CM
    { id: '8', type: 'attacker', position: { x: 46, y: 28 } },   // LCM
    { id: '9', type: 'attacker', position: { x: 56, y: 58 } },   // RW
    { id: '10', type: 'attacker', position: { x: 56, y: 40 } },  // ST
    { id: '11', type: 'attacker', position: { x: 56, y: 22 } },  // LW
    // Away team
    { id: 'd1', type: 'defender', position: { x: 108, y: 40 } },
    { id: 'd2', type: 'defender', position: { x: 92, y: 16 } },
    { id: 'd3', type: 'defender', position: { x: 92, y: 32 } },
    { id: 'd4', type: 'defender', position: { x: 92, y: 48 } },
    { id: 'd5', type: 'defender', position: { x: 92, y: 64 } },
    { id: 'd6', type: 'defender', position: { x: 74, y: 28 } },
    { id: 'd7', type: 'defender', position: { x: 74, y: 40 } },
    { id: 'd8', type: 'defender', position: { x: 74, y: 52 } },
    { id: 'd9', type: 'defender', position: { x: 64, y: 22 } },
    { id: 'd10', type: 'defender', position: { x: 64, y: 40 } },
    { id: 'd11', type: 'defender', position: { x: 64, y: 58 } },
  ],
};

export const FORMATION_4_4_2: Preset = {
  name: '4-4-2',
  ballCarrier: '10',
  players: [
    // Home team
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },   // GK
    { id: '2', type: 'attacker', position: { x: 28, y: 64 } },   // RB
    { id: '3', type: 'attacker', position: { x: 28, y: 48 } },   // CB
    { id: '4', type: 'attacker', position: { x: 28, y: 32 } },   // CB
    { id: '5', type: 'attacker', position: { x: 28, y: 16 } },   // LB
    { id: '6', type: 'attacker', position: { x: 46, y: 60 } },   // RM
    { id: '7', type: 'attacker', position: { x: 46, y: 46 } },   // RCM
    { id: '8', type: 'attacker', position: { x: 46, y: 34 } },   // LCM
    { id: '9', type: 'attacker', position: { x: 46, y: 20 } },   // LM
    { id: '10', type: 'attacker', position: { x: 56, y: 48 } },  // ST
    { id: '11', type: 'attacker', position: { x: 56, y: 32 } },  // ST
    // Away team
    { id: 'd1', type: 'defender', position: { x: 108, y: 40 } },
    { id: 'd2', type: 'defender', position: { x: 92, y: 16 } },
    { id: 'd3', type: 'defender', position: { x: 92, y: 32 } },
    { id: 'd4', type: 'defender', position: { x: 92, y: 48 } },
    { id: 'd5', type: 'defender', position: { x: 92, y: 64 } },
    { id: 'd6', type: 'defender', position: { x: 74, y: 20 } },
    { id: 'd7', type: 'defender', position: { x: 74, y: 34 } },
    { id: 'd8', type: 'defender', position: { x: 74, y: 46 } },
    { id: 'd9', type: 'defender', position: { x: 74, y: 60 } },
    { id: 'd10', type: 'defender', position: { x: 64, y: 48 } },
    { id: 'd11', type: 'defender', position: { x: 64, y: 32 } },
  ],
};

export const FORMATION_3_5_2: Preset = {
  name: '3-5-2',
  ballCarrier: '8',
  players: [
    // Home team
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },   // GK
    { id: '2', type: 'attacker', position: { x: 28, y: 56 } },   // RCB
    { id: '3', type: 'attacker', position: { x: 28, y: 40 } },   // CB
    { id: '4', type: 'attacker', position: { x: 28, y: 24 } },   // LCB
    { id: '5', type: 'attacker', position: { x: 46, y: 68 } },   // RWB
    { id: '6', type: 'attacker', position: { x: 46, y: 52 } },   // RM
    { id: '7', type: 'attacker', position: { x: 46, y: 40 } },   // CM
    { id: '8', type: 'attacker', position: { x: 46, y: 28 } },   // LM
    { id: '9', type: 'attacker', position: { x: 46, y: 12 } },   // LWB
    { id: '10', type: 'attacker', position: { x: 56, y: 48 } },  // ST
    { id: '11', type: 'attacker', position: { x: 56, y: 32 } },  // ST
    // Away team
    { id: 'd1', type: 'defender', position: { x: 108, y: 40 } },
    { id: 'd2', type: 'defender', position: { x: 92, y: 24 } },
    { id: 'd3', type: 'defender', position: { x: 92, y: 40 } },
    { id: 'd4', type: 'defender', position: { x: 92, y: 56 } },
    { id: 'd5', type: 'defender', position: { x: 74, y: 12 } },
    { id: 'd6', type: 'defender', position: { x: 74, y: 28 } },
    { id: 'd7', type: 'defender', position: { x: 74, y: 40 } },
    { id: 'd8', type: 'defender', position: { x: 74, y: 52 } },
    { id: 'd9', type: 'defender', position: { x: 74, y: 68 } },
    { id: 'd10', type: 'defender', position: { x: 64, y: 32 } },
    { id: 'd11', type: 'defender', position: { x: 64, y: 48 } },
  ],
};

export const FORMATION_4_2_3_1: Preset = {
  name: '4-2-3-1',
  ballCarrier: '10',
  players: [
    // Home team
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },   // GK
    { id: '2', type: 'attacker', position: { x: 28, y: 64 } },   // RB
    { id: '3', type: 'attacker', position: { x: 28, y: 48 } },   // CB
    { id: '4', type: 'attacker', position: { x: 28, y: 32 } },   // CB
    { id: '5', type: 'attacker', position: { x: 28, y: 16 } },   // LB
    { id: '6', type: 'attacker', position: { x: 42, y: 48 } },   // CDM
    { id: '7', type: 'attacker', position: { x: 42, y: 32 } },   // CDM
    { id: '8', type: 'attacker', position: { x: 52, y: 56 } },   // RAM
    { id: '9', type: 'attacker', position: { x: 52, y: 40 } },   // CAM
    { id: '10', type: 'attacker', position: { x: 56, y: 40 } },  // ST
    { id: '11', type: 'attacker', position: { x: 52, y: 24 } },  // LAM
    // Away team
    { id: 'd1', type: 'defender', position: { x: 108, y: 40 } },
    { id: 'd2', type: 'defender', position: { x: 92, y: 16 } },
    { id: 'd3', type: 'defender', position: { x: 92, y: 32 } },
    { id: 'd4', type: 'defender', position: { x: 92, y: 48 } },
    { id: 'd5', type: 'defender', position: { x: 92, y: 64 } },
    { id: 'd6', type: 'defender', position: { x: 78, y: 32 } },
    { id: 'd7', type: 'defender', position: { x: 78, y: 48 } },
    { id: 'd8', type: 'defender', position: { x: 68, y: 24 } },
    { id: 'd9', type: 'defender', position: { x: 68, y: 40 } },
    { id: 'd10', type: 'defender', position: { x: 64, y: 40 } },
    { id: 'd11', type: 'defender', position: { x: 68, y: 56 } },
  ],
};

export const FORMATION_3_4_3: Preset = {
  name: '3-4-3',
  ballCarrier: '10',
  players: [
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },
    { id: '2', type: 'attacker', position: { x: 28, y: 56 } },
    { id: '3', type: 'attacker', position: { x: 28, y: 40 } },
    { id: '4', type: 'attacker', position: { x: 28, y: 24 } },
    { id: '5', type: 'attacker', position: { x: 46, y: 64 } },
    { id: '6', type: 'attacker', position: { x: 46, y: 48 } },
    { id: '7', type: 'attacker', position: { x: 46, y: 32 } },
    { id: '8', type: 'attacker', position: { x: 46, y: 16 } },
    { id: '9', type: 'attacker', position: { x: 56, y: 58 } },
    { id: '10', type: 'attacker', position: { x: 56, y: 40 } },
    { id: '11', type: 'attacker', position: { x: 56, y: 22 } },
    { id: 'd1', type: 'defender', position: { x: 108, y: 40 } },
    { id: 'd2', type: 'defender', position: { x: 92, y: 24 } },
    { id: 'd3', type: 'defender', position: { x: 92, y: 40 } },
    { id: 'd4', type: 'defender', position: { x: 92, y: 56 } },
    { id: 'd5', type: 'defender', position: { x: 74, y: 16 } },
    { id: 'd6', type: 'defender', position: { x: 74, y: 32 } },
    { id: 'd7', type: 'defender', position: { x: 74, y: 48 } },
    { id: 'd8', type: 'defender', position: { x: 74, y: 64 } },
    { id: 'd9', type: 'defender', position: { x: 64, y: 22 } },
    { id: 'd10', type: 'defender', position: { x: 64, y: 40 } },
    { id: 'd11', type: 'defender', position: { x: 64, y: 58 } },
  ],
};

export const FORMATION_5_3_2: Preset = {
  name: '5-3-2',
  ballCarrier: '8',
  players: [
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },
    { id: '2', type: 'attacker', position: { x: 28, y: 68 } },
    { id: '3', type: 'attacker', position: { x: 28, y: 52 } },
    { id: '4', type: 'attacker', position: { x: 28, y: 40 } },
    { id: '5', type: 'attacker', position: { x: 28, y: 28 } },
    { id: '6', type: 'attacker', position: { x: 28, y: 12 } },
    { id: '7', type: 'attacker', position: { x: 46, y: 48 } },
    { id: '8', type: 'attacker', position: { x: 46, y: 36 } },
    { id: '9', type: 'attacker', position: { x: 46, y: 24 } },
    { id: '10', type: 'attacker', position: { x: 56, y: 46 } },
    { id: '11', type: 'attacker', position: { x: 56, y: 34 } },
    { id: 'd1', type: 'defender', position: { x: 108, y: 40 } },
    { id: 'd2', type: 'defender', position: { x: 92, y: 16 } },
    { id: 'd3', type: 'defender', position: { x: 92, y: 28 } },
    { id: 'd4', type: 'defender', position: { x: 92, y: 44 } },
    { id: 'd5', type: 'defender', position: { x: 92, y: 60 } },
    { id: 'd6', type: 'defender', position: { x: 74, y: 12 } },
    { id: 'd7', type: 'defender', position: { x: 74, y: 28 } },
    { id: 'd8', type: 'defender', position: { x: 74, y: 44 } },
    { id: 'd9', type: 'defender', position: { x: 74, y: 60 } },
    { id: 'd10', type: 'defender', position: { x: 64, y: 32 } },
    { id: 'd11', type: 'defender', position: { x: 64, y: 48 } },
  ],
};

export const FORMATION_4_1_4_1: Preset = {
  name: '4-1-4-1',
  ballCarrier: '9',
  players: [
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },
    { id: '2', type: 'attacker', position: { x: 28, y: 64 } },
    { id: '3', type: 'attacker', position: { x: 28, y: 48 } },
    { id: '4', type: 'attacker', position: { x: 28, y: 32 } },
    { id: '5', type: 'attacker', position: { x: 28, y: 16 } },
    { id: '6', type: 'attacker', position: { x: 42, y: 40 } },
    { id: '7', type: 'attacker', position: { x: 50, y: 56 } },
    { id: '8', type: 'attacker', position: { x: 50, y: 44 } },
    { id: '9', type: 'attacker', position: { x: 50, y: 28 } },
    { id: '10', type: 'attacker', position: { x: 50, y: 16 } },
    { id: '11', type: 'attacker', position: { x: 56, y: 36 } },
    { id: 'd1', type: 'defender', position: { x: 108, y: 40 } },
    { id: 'd2', type: 'defender', position: { x: 92, y: 16 } },
    { id: 'd3', type: 'defender', position: { x: 92, y: 32 } },
    { id: 'd4', type: 'defender', position: { x: 92, y: 48 } },
    { id: 'd5', type: 'defender', position: { x: 92, y: 64 } },
    { id: 'd6', type: 'defender', position: { x: 78, y: 36 } },
    { id: 'd7', type: 'defender', position: { x: 68, y: 52 } },
    { id: 'd8', type: 'defender', position: { x: 68, y: 40 } },
    { id: 'd9', type: 'defender', position: { x: 68, y: 28 } },
    { id: 'd10', type: 'defender', position: { x: 64, y: 52 } },
    { id: 'd11', type: 'defender', position: { x: 64, y: 28 } },
  ],
};

export const FORMATION_4_3_1_2: Preset = {
  name: '4-3-1-2',
  ballCarrier: '10',
  players: [
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },
    { id: '2', type: 'attacker', position: { x: 28, y: 64 } },
    { id: '3', type: 'attacker', position: { x: 28, y: 48 } },
    { id: '4', type: 'attacker', position: { x: 28, y: 32 } },
    { id: '5', type: 'attacker', position: { x: 28, y: 16 } },
    { id: '6', type: 'attacker', position: { x: 46, y: 52 } },
    { id: '7', type: 'attacker', position: { x: 46, y: 40 } },
    { id: '8', type: 'attacker', position: { x: 46, y: 28 } },
    { id: '9', type: 'attacker', position: { x: 52, y: 40 } },
    { id: '10', type: 'attacker', position: { x: 56, y: 46 } },
    { id: '11', type: 'attacker', position: { x: 56, y: 34 } },
    { id: 'd1', type: 'defender', position: { x: 108, y: 40 } },
    { id: 'd2', type: 'defender', position: { x: 92, y: 16 } },
    { id: 'd3', type: 'defender', position: { x: 92, y: 32 } },
    { id: 'd4', type: 'defender', position: { x: 92, y: 48 } },
    { id: 'd5', type: 'defender', position: { x: 92, y: 64 } },
    { id: 'd6', type: 'defender', position: { x: 78, y: 24 } },
    { id: 'd7', type: 'defender', position: { x: 78, y: 40 } },
    { id: 'd8', type: 'defender', position: { x: 78, y: 56 } },
    { id: 'd9', type: 'defender', position: { x: 64, y: 46 } },
    { id: 'd10', type: 'defender', position: { x: 64, y: 34 } },
    { id: 'd11', type: 'defender', position: { x: 60, y: 40 } },
  ],
};

export const FORMATION_4_4_2_DIAMOND: Preset = {
  name: '4-4-2 (Diamond)',
  ballCarrier: '9',
  players: [
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },
    { id: '2', type: 'attacker', position: { x: 28, y: 64 } },
    { id: '3', type: 'attacker', position: { x: 28, y: 48 } },
    { id: '4', type: 'attacker', position: { x: 28, y: 32 } },
    { id: '5', type: 'attacker', position: { x: 28, y: 16 } },
    { id: '6', type: 'attacker', position: { x: 42, y: 40 } },
    { id: '7', type: 'attacker', position: { x: 48, y: 46 } },
    { id: '8', type: 'attacker', position: { x: 48, y: 34 } },
    { id: '9', type: 'attacker', position: { x: 52, y: 40 } },
    { id: '10', type: 'attacker', position: { x: 56, y: 46 } },
    { id: '11', type: 'attacker', position: { x: 56, y: 34 } },
    { id: 'd1', type: 'defender', position: { x: 108, y: 40 } },
    { id: 'd2', type: 'defender', position: { x: 92, y: 16 } },
    { id: 'd3', type: 'defender', position: { x: 92, y: 32 } },
    { id: 'd4', type: 'defender', position: { x: 92, y: 48 } },
    { id: 'd5', type: 'defender', position: { x: 92, y: 64 } },
    { id: 'd6', type: 'defender', position: { x: 78, y: 36 } },
    { id: 'd7', type: 'defender', position: { x: 68, y: 52 } },
    { id: 'd8', type: 'defender', position: { x: 68, y: 28 } },
    { id: 'd9', type: 'defender', position: { x: 64, y: 46 } },
    { id: 'd10', type: 'defender', position: { x: 64, y: 34 } },
    { id: 'd11', type: 'defender', position: { x: 60, y: 40 } },
  ],
};

// Attacking Situations

export const CORNER_LEFT: Preset = {
  name: 'Corner (Left)',
  ballCarrier: '11',
  players: [
    // Home team - attacking
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },   // GK stays back
    { id: '2', type: 'attacker', position: { x: 60, y: 40 } },   // Stay back
    { id: '3', type: 'attacker', position: { x: 100, y: 48 } },  // In box
    { id: '4', type: 'attacker', position: { x: 96, y: 40 } },   // In box
    { id: '5', type: 'attacker', position: { x: 100, y: 32 } },  // In box
    { id: '6', type: 'attacker', position: { x: 92, y: 52 } },   // In box
    { id: '7', type: 'attacker', position: { x: 88, y: 40 } },   // Edge of box
    { id: '8', type: 'attacker', position: { x: 92, y: 28 } },   // In box
    { id: '9', type: 'attacker', position: { x: 96, y: 56 } },   // Near post
    { id: '10', type: 'attacker', position: { x: 96, y: 24 } },  // Far post
    { id: '11', type: 'attacker', position: { x: 119, y: 1 } },  // Corner taker (bottom-right corner)
    // Away team - defending
    { id: 'd1', type: 'defender', position: { x: 116, y: 40 } }, // GK on line
    { id: 'd2', type: 'defender', position: { x: 114, y: 28 } }, // Near post
    { id: 'd3', type: 'defender', position: { x: 100, y: 36 } }, // Marking
    { id: 'd4', type: 'defender', position: { x: 100, y: 44 } }, // Marking
    { id: 'd5', type: 'defender', position: { x: 114, y: 52 } }, // Far post
    { id: 'd6', type: 'defender', position: { x: 96, y: 32 } },  // Marking
    { id: 'd7', type: 'defender', position: { x: 96, y: 48 } },  // Marking
    { id: 'd8', type: 'defender', position: { x: 92, y: 40 } },  // Marking
    { id: 'd9', type: 'defender', position: { x: 88, y: 28 } },  // Edge
    { id: 'd10', type: 'defender', position: { x: 88, y: 52 } }, // Edge
    { id: 'd11', type: 'defender', position: { x: 60, y: 40 } }, // Midfield
  ],
};

export const CORNER_RIGHT: Preset = {
  name: 'Corner (Right)',
  ballCarrier: '7',
  players: [
    // Home team - attacking
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },   // GK stays back
    { id: '2', type: 'attacker', position: { x: 60, y: 40 } },   // Stay back
    { id: '3', type: 'attacker', position: { x: 100, y: 48 } },  // In box
    { id: '4', type: 'attacker', position: { x: 96, y: 40 } },   // In box
    { id: '5', type: 'attacker', position: { x: 100, y: 32 } },  // In box
    { id: '6', type: 'attacker', position: { x: 92, y: 52 } },   // In box
    { id: '7', type: 'attacker', position: { x: 119, y: 79 } },  // Corner taker (top-right corner)
    { id: '8', type: 'attacker', position: { x: 92, y: 28 } },   // In box
    { id: '9', type: 'attacker', position: { x: 96, y: 56 } },   // Near post
    { id: '10', type: 'attacker', position: { x: 96, y: 24 } },  // Far post
    { id: '11', type: 'attacker', position: { x: 88, y: 40 } },  // Edge of box
    // Away team - defending
    { id: 'd1', type: 'defender', position: { x: 116, y: 40 } },
    { id: 'd2', type: 'defender', position: { x: 114, y: 28 } },
    { id: 'd3', type: 'defender', position: { x: 100, y: 36 } },
    { id: 'd4', type: 'defender', position: { x: 100, y: 44 } },
    { id: 'd5', type: 'defender', position: { x: 114, y: 52 } },
    { id: 'd6', type: 'defender', position: { x: 96, y: 32 } },
    { id: 'd7', type: 'defender', position: { x: 96, y: 48 } },
    { id: 'd8', type: 'defender', position: { x: 92, y: 40 } },
    { id: 'd9', type: 'defender', position: { x: 88, y: 28 } },
    { id: 'd10', type: 'defender', position: { x: 88, y: 52 } },
    { id: 'd11', type: 'defender', position: { x: 60, y: 40 } },
  ],
};

export const FREE_KICK_EDGE: Preset = {
  name: 'Free Kick (Edge of Box)',
  ballCarrier: '10',
  players: [
    // Home team - attacking
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },   // GK
    { id: '2', type: 'attacker', position: { x: 60, y: 52 } },   // Midfield
    { id: '3', type: 'attacker', position: { x: 60, y: 28 } },   // Midfield
    { id: '4', type: 'attacker', position: { x: 50, y: 40 } },   // Midfield
    { id: '5', type: 'attacker', position: { x: 96, y: 24 } },   // In box (far post)
    { id: '6', type: 'attacker', position: { x: 100, y: 32 } },  // In box
    { id: '7', type: 'attacker', position: { x: 96, y: 56 } },   // In box (near post)
    { id: '8', type: 'attacker', position: { x: 88, y: 40 } },   // Edge of box
    { id: '9', type: 'attacker', position: { x: 100, y: 48 } },  // In box
    { id: '10', type: 'attacker', position: { x: 102, y: 46 } }, // Free kick taker
    { id: '11', type: 'attacker', position: { x: 101, y: 44 } }, // Dummy runner
    // Away team - defending
    { id: 'd1', type: 'defender', position: { x: 116, y: 40 } }, // GK
    { id: 'd2', type: 'defender', position: { x: 110, y: 40 } }, // Wall
    { id: 'd3', type: 'defender', position: { x: 110, y: 43 } }, // Wall
    { id: 'd4', type: 'defender', position: { x: 110, y: 46 } }, // Wall
    { id: 'd5', type: 'defender', position: { x: 110, y: 49 } }, // Wall
    { id: 'd6', type: 'defender', position: { x: 96, y: 32 } },  // Marking
    { id: 'd7', type: 'defender', position: { x: 96, y: 48 } },  // Marking
    { id: 'd8', type: 'defender', position: { x: 100, y: 36 } }, // Marking
    { id: 'd9', type: 'defender', position: { x: 100, y: 52 } }, // Marking
    { id: 'd10', type: 'defender', position: { x: 92, y: 40 } }, // Marking
    { id: 'd11', type: 'defender', position: { x: 70, y: 40 } }, // Midfield
  ],
};

export const COUNTER_ATTACK: Preset = {
  name: 'Counter Attack',
  ballCarrier: '10',
  players: [
    // Home team - counter attacking
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },   // GK
    { id: '2', type: 'attacker', position: { x: 32, y: 60 } },   // Pushing up
    { id: '3', type: 'attacker', position: { x: 28, y: 48 } },   // CB
    { id: '4', type: 'attacker', position: { x: 28, y: 32 } },   // CB
    { id: '5', type: 'attacker', position: { x: 32, y: 20 } },   // Pushing up
    { id: '6', type: 'attacker', position: { x: 48, y: 40 } },   // Midfielder with space
    { id: '7', type: 'attacker', position: { x: 76, y: 64 } },   // Wing run
    { id: '8', type: 'attacker', position: { x: 56, y: 40 } },   // Advancing
    { id: '9', type: 'attacker', position: { x: 88, y: 48 } },   // Forward run
    { id: '10', type: 'attacker', position: { x: 68, y: 40 } },  // Ball carrier (midfield)
    { id: '11', type: 'attacker', position: { x: 76, y: 16 } },  // Wing run
    // Away team - caught out of position
    { id: 'd1', type: 'defender', position: { x: 116, y: 40 } }, // GK
    { id: 'd2', type: 'defender', position: { x: 80, y: 24 } },  // Recovering
    { id: 'd3', type: 'defender', position: { x: 88, y: 36 } },  // Recovering
    { id: 'd4', type: 'defender', position: { x: 88, y: 44 } },  // Recovering
    { id: 'd5', type: 'defender', position: { x: 80, y: 56 } },  // Recovering
    { id: 'd6', type: 'defender', position: { x: 52, y: 32 } },  // Caught high
    { id: 'd7', type: 'defender', position: { x: 52, y: 48 } },  // Caught high
    { id: 'd8', type: 'defender', position: { x: 44, y: 40 } },  // Caught high
    { id: 'd9', type: 'defender', position: { x: 36, y: 24 } },  // Caught very high
    { id: 'd10', type: 'defender', position: { x: 36, y: 56 } }, // Caught very high
    { id: 'd11', type: 'defender', position: { x: 72, y: 40 } }, // Chasing back
  ],
};

export const PENALTY_KICK: Preset = {
  name: 'Penalty Kick',
  ballCarrier: '10',
  players: [
    // Home team - taking penalty
    { id: '1', type: 'attacker', position: { x: 12, y: 40 } },   // GK
    { id: '2', type: 'attacker', position: { x: 88, y: 64 } },   // Outside box
    { id: '3', type: 'attacker', position: { x: 88, y: 52 } },   // Outside box
    { id: '4', type: 'attacker', position: { x: 88, y: 28 } },   // Outside box
    { id: '5', type: 'attacker', position: { x: 88, y: 16 } },   // Outside box
    { id: '6', type: 'attacker', position: { x: 76, y: 40 } },   // Back
    { id: '7', type: 'attacker', position: { x: 64, y: 52 } },   // Back
    { id: '8', type: 'attacker', position: { x: 64, y: 28 } },   // Back
    { id: '9', type: 'attacker', position: { x: 50, y: 40 } },   // Back
    { id: '10', type: 'attacker', position: { x: 108, y: 40 } }, // Penalty taker
    { id: '11', type: 'attacker', position: { x: 88, y: 40 } },  // Ready for rebound
    // Away team - defending penalty
    { id: 'd1', type: 'defender', position: { x: 120, y: 40 } }, // GK on line
    { id: 'd2', type: 'defender', position: { x: 88, y: 70 } },  // Outside box
    { id: 'd3', type: 'defender', position: { x: 88, y: 58 } },  // Outside box
    { id: 'd4', type: 'defender', position: { x: 88, y: 22 } },  // Outside box
    { id: 'd5', type: 'defender', position: { x: 88, y: 10 } },  // Outside box
    { id: 'd6', type: 'defender', position: { x: 76, y: 40 } },  // Back
    { id: 'd7', type: 'defender', position: { x: 64, y: 52 } },  // Back
    { id: 'd8', type: 'defender', position: { x: 64, y: 28 } },  // Back
    { id: 'd9', type: 'defender', position: { x: 50, y: 40 } },  // Back
    { id: 'd10', type: 'defender', position: { x: 88, y: 46 } }, // Outside box
    { id: 'd11', type: 'defender', position: { x: 88, y: 34 } }, // Outside box
  ],
};

export const FORMATION_PRESETS = [
  FORMATION_4_3_3,
  FORMATION_4_4_2,
  FORMATION_3_5_2,
  FORMATION_4_2_3_1,
];

export const SITUATION_PRESETS = [
  CORNER_LEFT,
  CORNER_RIGHT,
  FREE_KICK_EDGE,
  COUNTER_ATTACK,
  PENALTY_KICK,
];
