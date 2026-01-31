import type { Player } from './types';

// Backend API configuration
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000';

interface BackendResponse {
  attackers: Array<{ x: number; y: number; id: string }>;
  defenders: Array<{ x: number; y: number; id: string }>;
  ball_id: string;
}

interface LLMResponse {
  ballCarrier: string;
  players: Player[];
}

export async function generateSituation(situation: string): Promise<LLMResponse> {
  // Send situation to backend
  const response = await fetch(
    `${BACKEND_URL}/generate-positions?situation=${encodeURIComponent(situation)}`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }));
    throw new Error(error.message || `Backend error: ${response.status}`);
  }

  const data: BackendResponse = await response.json();

  // Validate response structure
  if (!Array.isArray(data.attackers) || !Array.isArray(data.defenders) || !data.ball_id) {
    throw new Error('Invalid response structure from backend');
  }

  if (data.attackers.length !== 11 || data.defenders.length !== 11) {
    throw new Error(`Invalid number of players: ${data.attackers.length} attackers, ${data.defenders.length} defenders`);
  }

  // Convert backend format to our internal format
  const players: Player[] = [
    ...data.attackers.map(a => ({
      id: a.id,
      type: 'attacker' as const,
      position: {
        x: Math.max(0, Math.min(120, a.x)),
        y: Math.max(0, Math.min(80, a.y)),
      },
    })),
    ...data.defenders.map(d => ({
      id: d.id,
      type: 'defender' as const,
      position: {
        x: Math.max(0, Math.min(120, d.x)),
        y: Math.max(0, Math.min(80, d.y)),
      },
    })),
  ];

  return {
    ballCarrier: data.ball_id,
    players,
  };
}
