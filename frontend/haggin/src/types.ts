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
