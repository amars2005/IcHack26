import { useState, useCallback } from 'react';
import type { PitchPlayer, Keeper, PitchState } from '../types';

const createInitialAttackers = (): PitchPlayer[] => [
  { id: '2', x: 28, y: 64, team: 1 },   // RB
  { id: '3', x: 28, y: 48, team: 1 },   // RCB
  { id: '4', x: 28, y: 32, team: 1 },   // LCB
  { id: '5', x: 28, y: 16, team: 1 },   // LB
  { id: '6', x: 46, y: 52, team: 1 },   // RCM
  { id: '7', x: 46, y: 40, team: 1 },   // CM
  { id: '8', x: 46, y: 28, team: 1 },   // LCM
  { id: '9', x: 56, y: 58, team: 1 },   // RW
  { id: '10', x: 56, y: 40, team: 1 },  // ST
  { id: '11', x: 56, y: 22, team: 1 },  // LW
];

const createInitialDefenders = (): PitchPlayer[] => [
  { id: 'd2', x: 92, y: 16, team: 0 },   // LB
  { id: 'd3', x: 92, y: 32, team: 0 },   // LCB
  { id: 'd4', x: 92, y: 48, team: 0 },   // RCB
  { id: 'd5', x: 92, y: 64, team: 0 },   // RB
  { id: 'd6', x: 74, y: 28, team: 0 },   // LCM
  { id: 'd7', x: 74, y: 40, team: 0 },   // CM
  { id: 'd8', x: 74, y: 52, team: 0 },   // RCM
  { id: 'd9', x: 64, y: 22, team: 0 },   // LW
  { id: 'd10', x: 64, y: 40, team: 0 },  // ST
  { id: 'd11', x: 64, y: 58, team: 0 },  // RW
];

const createInitialKeepers = (): Keeper[] => [
  { id: '1', x: 12, y: 40, team: 1 },    // Attacking team keeper
  { id: 'd1', x: 108, y: 40, team: 0 },  // Defending team keeper
];

const createInitialState = (): PitchState => ({
  attackers: createInitialAttackers(),
  defenders: createInitialDefenders(),
  keepers: createInitialKeepers(),
  ballId: '9', // Ball starts with RW
});

export const usePitchState = () => {
  const [pitchState, setPitchState] = useState<PitchState>(createInitialState);

  const updatePlayerPosition = useCallback((id: string, x: number, y: number) => {
    setPitchState((prev) => {
      // Check attackers
      const attackerIndex = prev.attackers.findIndex((p) => p.id === id);
      if (attackerIndex !== -1) {
        const newAttackers = [...prev.attackers];
        newAttackers[attackerIndex] = { ...newAttackers[attackerIndex], x, y };
        return { ...prev, attackers: newAttackers };
      }

      // Check defenders
      const defenderIndex = prev.defenders.findIndex((p) => p.id === id);
      if (defenderIndex !== -1) {
        const newDefenders = [...prev.defenders];
        newDefenders[defenderIndex] = { ...newDefenders[defenderIndex], x, y };
        return { ...prev, defenders: newDefenders };
      }

      // Check keepers
      const keeperIndex = prev.keepers.findIndex((k) => k.id === id);
      if (keeperIndex !== -1) {
        const newKeepers = [...prev.keepers];
        newKeepers[keeperIndex] = { ...newKeepers[keeperIndex], x, y };
        return { ...prev, keepers: newKeepers };
      }

      return prev;
    });
  }, []);

  const setBallId = useCallback((id: string) => {
    setPitchState((prev) => ({ ...prev, ballId: id }));
  }, []);

  const setPositions = useCallback((
    attackers: Array<{ id: string; x: number; y: number }>,
    defenders: Array<{ id: string; x: number; y: number }>,
    ballId: string
  ) => {
    // Separate keepers from outfield players
    const attackerKeeper = attackers.find((a) => a.id === '1');
    const defenderKeeper = defenders.find((d) => d.id === 'd1');
    
    const outfieldAttackers = attackers.filter((a) => a.id !== '1');
    const outfieldDefenders = defenders.filter((d) => d.id !== 'd1');

    setPitchState((prev) => ({
      ...prev,
      attackers: outfieldAttackers.map((a) => ({ ...a, team: 1 as const })),
      defenders: outfieldDefenders.map((d) => ({ ...d, team: 0 as const })),
      keepers: [
        attackerKeeper 
          ? { ...attackerKeeper, team: 1 as const }
          : prev.keepers.find((k) => k.team === 1) || { id: '1', x: 12, y: 40, team: 1 as const },
        defenderKeeper
          ? { ...defenderKeeper, team: 0 as const }
          : prev.keepers.find((k) => k.team === 0) || { id: 'd1', x: 108, y: 40, team: 0 as const },
      ],
      ballId,
    }));
  }, []);

  const resetPositions = useCallback(() => {
    setPitchState(createInitialState());
  }, []);

  const getPlayerById = useCallback((id: string): PitchPlayer | Keeper | undefined => {
    return (
      pitchState.attackers.find((p) => p.id === id) ||
      pitchState.defenders.find((p) => p.id === id) ||
      pitchState.keepers.find((k) => k.id === id)
    );
  }, [pitchState]);

  const getAllPlayers = useCallback((): (PitchPlayer | Keeper)[] => {
    return [...pitchState.attackers, ...pitchState.defenders, ...pitchState.keepers];
  }, [pitchState]);

  // Format data for backend API
  const getApiPayload = useCallback(() => {
    return {
      attackers: pitchState.attackers.map(({ id, x, y, team }) => ({ id, x, y, team })),
      defenders: pitchState.defenders.map(({ id, x, y, team }) => ({ id, x, y, team })),
      keepers: pitchState.keepers.map(({ id, x, y, team }) => ({ id, x, y, team })),
      ball_id: pitchState.ballId,
    };
  }, [pitchState]);

  return {
    pitchState,
    updatePlayerPosition,
    setBallId,
    setPositions,
    resetPositions,
    getPlayerById,
    getAllPlayers,
    getApiPayload,
  };
};
