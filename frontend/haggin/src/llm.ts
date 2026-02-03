import type { Player } from './types';

// Backend API configuration
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5001';

interface BackendResponse {
  attackers: Array<{ x: number; y: number; id: string }>;
  defenders: Array<{ x: number; y: number; id: string }>;
  ball_id: string;
}

interface LLMResponse {
  ballCarrier: string;
  players: Player[];
}

export class AIError extends Error {
  code: string;
  constructor(code: string, message?: string) {
    super(message || code);
    this.name = 'AIError';
    this.code = code;
  }
}

export async function generateSituation(situation: string): Promise<LLMResponse> {
  // Send situation to backend
  const response = await fetch(
    `$/api/generate-positions?situation=${encodeURIComponent(situation)}`,
    {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Unknown error' }));
    // backend can return structured error like { error: 'inappropriate', message: '...' }
    if (error && error.error === 'inappropriate') {
      throw new AIError('INAPPROPRIATE_PROMPT', error.message || 'Prompt rejected by policy');
    }
    throw new AIError('BACKEND_ERROR', error.message || `Backend error: ${response.status}`);
  }

  const data: BackendResponse = await response.json();

  // Validate response structure
  if (!Array.isArray(data.attackers) || !Array.isArray(data.defenders) || !data.ball_id) {
    console.log('Received data:', data);
    throw new AIError('INVALID_RESPONSE', 'Invalid response structure from backend');
  }

  if (data.attackers.length !== 11 || data.defenders.length !== 11) {
    throw new AIError('INVALID_RESPONSE', `Invalid number of players: ${data.attackers.length} attackers, ${data.defenders.length} defenders`);
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

// Fetch simple player metrics (xG, xT, etc) from backend for a given player id.
// Backend should return a JSON object like: { xG: 0.02, xT: 0.01, shots: 1 }
export async function fetchPlayerMetrics(playerId: string): Promise<Record<string, number>> {
  const res = await fetch(`$/api/player-metrics?playerId=${encodeURIComponent(playerId)}`);
  if (!res.ok) {
    // return empty object on failure; caller can handle
    return {};
  }
  const json = await res.json().catch(() => ({}));
  // Ensure numeric values
  const out: Record<string, number> = {};
  if (json && typeof json === 'object') {
    for (const [k, v] of Object.entries(json)) {
      out[k] = typeof v === 'number' ? v : Number(v) || 0;
    }
  }
  return out;
}
