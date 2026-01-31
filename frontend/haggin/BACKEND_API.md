# Backend API Integration

The frontend sends requests to your backend to generate player positions using AI.

## API Endpoint

```
GET /generate-positions?situation=<description>
```

### Request

- **Method**: GET
- **Query Parameters**:
  - `situation` (string, required): Description of the attacking scenario

**Example**:
```
GET /generate-positions?situation=Fast%20counter-attack%20with%204%20attackers%20vs%203%20defenders
```

### Response Format

The backend should return JSON in this exact format:

```json
{
  "attackers": [
    {"x": 12, "y": 40, "id": "1"},
    {"x": 28, "y": 64, "id": "2"},
    {"x": 28, "y": 48, "id": "3"},
    {"x": 28, "y": 32, "id": "4"},
    {"x": 28, "y": 16, "id": "5"},
    {"x": 46, "y": 52, "id": "6"},
    {"x": 46, "y": 40, "id": "7"},
    {"x": 46, "y": 28, "id": "8"},
    {"x": 56, "y": 58, "id": "9"},
    {"x": 56, "y": 40, "id": "10"},
    {"x": 56, "y": 22, "id": "11"}
  ],
  "defenders": [
    {"x": 108, "y": 40, "id": "d1"},
    {"x": 92, "y": 16, "id": "d2"},
    {"x": 92, "y": 32, "id": "d3"},
    {"x": 92, "y": 48, "id": "d4"},
    {"x": 92, "y": 64, "id": "d5"},
    {"x": 74, "y": 28, "id": "d6"},
    {"x": 74, "y": 40, "id": "d7"},
    {"x": 74, "y": 52, "id": "d8"},
    {"x": 64, "y": 22, "id": "d9"},
    {"x": 64, "y": 40, "id": "d10"},
    {"x": 64, "y": 58, "id": "d11"}
  ],
  "ball_id": "9"
}
```

### Response Fields

- **attackers** (array, required): Array of 11 attacker positions
  - `x` (number): X coordinate (0-120 meters)
  - `y` (number): Y coordinate (0-80 meters)
  - `id` (string): Player ID ("1" through "11")

- **defenders** (array, required): Array of 11 defender positions
  - `x` (number): X coordinate (0-120 meters)
  - `y` (number): Y coordinate (0-80 meters)
  - `id` (string): Player ID ("d1" through "d11")

- **ball_id** (string, required): ID of the attacker with the ball (e.g., "9")

### Pitch Coordinate System

```
(0,80) ─────────────────────── (120,80)
  │                               │
  │    Attacking Team (Red)       │
  │    ← ← ← Direction            │
  │                               │
  │         Halfway               │
  │         x = 60                │
  │                               │
  │    Defending Team (Blue)      │
  │                               │
(0,0) ──────────────────────── (120,0)
```

- Bottom-left corner: (0, 0)
- Top-right corner: (120, 80)
- Attacking team attacks from LEFT to RIGHT
- Right goal at x=120, y=40
- Left goal at x=0, y=40

### Player ID Conventions

**Attackers** (IDs "1" to "11"):
- "1": Goalkeeper
- "2"-"5": Defenders
- "6"-"8": Midfielders
- "9"-"11": Forwards

**Defenders** (IDs "d1" to "d11"):
- "d1": Goalkeeper
- "d2"-"d5": Defenders
- "d6"-"d8": Midfielders
- "d9"-"d11": Forwards

### Error Response

```json
{
  "error": "Error message here",
  "message": "Detailed error description"
}
```

HTTP status codes:
- 200: Success
- 400: Bad request (invalid situation parameter)
- 500: Server error (LLM failure, etc.)

## Frontend Configuration

Set the backend URL in `.env`:

```bash
VITE_BACKEND_URL=http://localhost:5000
```

Or it defaults to `http://localhost:5000` if not set.

## CORS Configuration

Your backend must allow requests from the frontend origin:

```python
# Example for Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5175"])
```

```javascript
// Example for Express
const cors = require('cors');
app.use(cors({ origin: 'http://localhost:5175' }));
```

## Testing the API

```bash
curl "http://localhost:5000/generate-positions?situation=counter%20attack"
```

Expected response should match the JSON format above with valid coordinates and IDs.

## Implementation Notes

1. **Validate input**: Check that situation parameter is not empty
2. **Call your LLM**: Use Claude/GPT with appropriate prompting
3. **Parse response**: Extract player positions from LLM output
4. **Validate output**: Ensure 11 attackers, 11 defenders, valid coordinates
5. **Return JSON**: Match the exact format specified above

## Example Backend Implementation (Python/Flask)

```python
from flask import Flask, request, jsonify
import anthropic

app = Flask(__name__)

@app.route('/generate-positions')
def generate_positions():
    situation = request.args.get('situation')

    if not situation:
        return jsonify({"error": "Missing situation parameter"}), 400

    # Call your LLM here
    # ... (use Claude/GPT with appropriate prompting)

    # Return formatted response
    return jsonify({
        "attackers": [...],  # 11 players with x, y, id
        "defenders": [...],  # 11 players with x, y, id
        "ball_id": "9"
    })

if __name__ == '__main__':
    app.run(port=5000)
```
