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

// xT calculation result from backend
export type XTResult = {
  xT?: number;
  error?: string;
  [key: string]: unknown;
};
