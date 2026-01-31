# Football Pitch Training Interface

A simple, maintainable drag-and-drop interface for football training with ML integration.

## Project Structure

```
src/
├── types.ts              # Type definitions (Player, Position, etc.)
├── constants.ts          # Configurable constants (colors, dimensions, initial setup)
├── components/
│   ├── Pitch.tsx        # Main pitch visualization
│   ├── Player.tsx       # Individual draggable player component
│   └── MenuBar.tsx      # Right-side control panel
└── App.tsx              # Main application logic
```

## Getting Started

```bash
npm install
npm run dev
```

## Coordinate System

- Origin (0, 0): Bottom-left corner
- Maximum (100, 60): Top-right corner
- Right goal position: (100, 30)
- Players can be dragged anywhere within these bounds

## Customization Guide

### Change Colors
Edit `src/constants.ts`:
```typescript
export const COLORS = {
  attacker: '#ef4444',      // Red - change to any hex color
  defender: '#3b82f6',      // Blue - change to any hex color
  pitch: '#22c55e',         // Green
  pitchBorder: '#16a34a',   // Dark green
  lines: '#ffffff',         // White
};
```

### Change Pitch Dimensions
Edit `src/constants.ts`:
```typescript
export const PITCH_WIDTH = 100;   // Change to desired width
export const PITCH_HEIGHT = 60;   // Change to desired height
```

### Change Player Size
Edit `src/constants.ts`:
```typescript
export const PLAYER_RADIUS = 8;   // Change radius in pixels
```

### Change Initial Player Positions
Edit `src/constants.ts`:
```typescript
export const INITIAL_PLAYERS = [
  { id: 'a1', type: 'attacker', position: { x: 30, y: 30 } },
  // Add or modify players here
];
```

### Modify Pitch Layout
Edit `src/components/Pitch.tsx` to add/remove field markings:
- Center circle, penalty areas, goals are all separate SVG elements
- Easy to comment out or add new markings

### Add New Player Types
1. Update `src/types.ts`:
```typescript
export type PlayerType = 'attacker' | 'defender' | 'goalkeeper';
```
2. Add color in `src/constants.ts`:
```typescript
export const COLORS = {
  attacker: '#ef4444',
  defender: '#3b82f6',
  goalkeeper: '#fbbf24',  // New type
  ...
};
```
3. Add button in `src/components/MenuBar.tsx`

### Export Player Positions
Access player positions via the `players` state in `App.tsx`:
```typescript
const exportPositions = () => {
  console.log(players.map(p => ({
    id: p.id,
    type: p.type,
    x: p.position.x,
    y: p.position.y,
  })));
};
```

## Features

- **Drag and Drop**: Click and drag any player to reposition
- **Add Players**: Use menu to add attackers or defenders
- **Reset**: Return all players to initial positions
- **Responsive**: Pitch scales to fit viewport
- **Menu Toggle**: Opens/closes to adjust pitch size

## Architecture Notes

- Pure React with TypeScript
- No external dependencies for drag-and-drop
- SVG-based rendering for crisp visuals at any scale
- Coordinate clamping keeps players within bounds
- State management via React hooks (easily upgradeable to context/redux if needed)
