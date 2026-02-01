export type Position = {
  x: number;
  y: number;
};

export type PlayerType = 'attacker' | 'defender';

export type GenerationStatus = 'idle' | 'loading' | 'success' | 'error';

export type Player = {
  id: string;
  type: PlayerType;
  position: Position;
};

// New types for pitch state management
export interface PitchPlayer {
  id: string;
  x: number;
  y: number;
  team: 0 | 1; // 0 = defender, 1 = attacker
}

export interface Keeper {
  id: string;
  x: number;
  y: number;
  team: 0 | 1; // 0 = defender, 1 = attacker
}

export interface PitchState {
  attackers: PitchPlayer[];
  defenders: PitchPlayer[];
  keepers: Keeper[];
  ballId: string | null;
}

// Heatmap data from NN model
export type HeatmapData = {
  heatmap: number[][];  // 2D array [rows][cols] of xT values
  x_coords: number[];   // X coordinates for each column
  y_coords: number[];   // Y coordinates for each row
};

// xT calculation result from backend
export type XTResult = {
  xT?: number;
  heatmap?: HeatmapData | null;
  error?: string;
  [key: string]: unknown;
};
