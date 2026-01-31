"""
Initialise app, and define GET route
"""
from flask import request, Flask
import requests
import json
import pandas as pd
import dotenv
dotenv.load_dotenv()

SYSTEM_PROMPT = """You are a tactical football/soccer analysis AI. Your task is to position 22 players (11 attacking, 11 defending) on a football pitch based on the described situation.

PITCH SPECIFICATIONS:
•⁠  ⁠Dimensions: 120m (width) x 80m (height)
•⁠  ⁠Coordinate system: (0,0) at bottom-left corner, (120,80) at top-right corner
•⁠  ⁠Attacking team (red) starts on the LEFT side and attacks toward the RIGHT
•⁠  ⁠Defending team (blue) starts on the RIGHT side and defends the RIGHT goal
•⁠  ⁠Right goal is at position x=120, y=40 (center of right edge)
•⁠  ⁠Left goal is at position x=0, y=40 (center of left edge)
•⁠  ⁠Halfway line is at x=60

PLAYER IDs:
•⁠  ⁠Attacking team: '1' through '11' (where '1' is goalkeeper, '2'-'5' are defenders, '6'-'8' are midfielders, '9'-'11' are forwards)
•⁠  ⁠Defending team: 'd1' through 'd11' (same position conventions)

TACTICAL GUIDELINES:
1.⁠ ⁠Position players realistically based on the described situation
2.⁠ ⁠Goalkeepers typically stay near their goal line (attacking GK at x≈12, defending GK at x≈108)
3.⁠ ⁠Consider the phase of play (attack, defense, transition, set piece)
4.⁠ ⁠Maintain realistic spacing between players (typically 5-15 meters)
5.⁠ ⁠For set pieces, position players according to standard tactics
6.⁠ ⁠Assign the ball carrier (ballCarrier) to the most appropriate attacking player

OUTPUT FORMAT:
EXPECTED FORMAT:
  {
    "attackers": [
      {"x": 12, "y": 40, "id": "1"},
      {"x": 28, "y": 64, "id": "2"},
      ... 11 total
    ],
    "defenders": [
      {"x": 108, "y": 40, "id": "d1"},
      {"x": 92, "y": 16, "id": "d2"},
      ... 11 total
    ],
    "ball_id": "9"
  }
"""


def start_app():
    app = Flask(__name__)

    @app.route("/", methods=["POST"])
    def predictions():
        if request.method == "POST":
            data = request.get_json()

            if not data:
                return {"error": "No data provided"}, 400

            attackers = data.get("attackers", [])
            defenders = data.get("defenders", [])
            print(attackers, defenders)

            attackers_positions = attackers[:1]
            defenders_positions = defenders[:2]

            ball_id = data["ball_id"]

            # now pass the data to the models for best action, xT and P(success)

            # generations
            import numpy as np
            xT = np.random.rand(12)
            p_success = np.random.rand(12)
            action = np.random.choice(
                ["pass", "carry", "shoot"], size=12
            )

            evaluations = {action[i]: {
                "xT": xT[i], "P(success)": p_success[i]} for i in range(12)}
            return json.dumps(evaluations, indent=4)

    @app.route("/generate-positions", methods=["GET"])
    def generate_positions():
        print("Generating positions...")
        if request.method == "GET":
            situation = request.args.get("situation", "")

        url = 'https://api.anthropic.com/v1/messages'

        headers = {
            'Content-Type': 'application/json',
            'x-api-key': dotenv.get_key(".env", "API_KEY"),
            'anthropic-version': '2023-06-01',
        }

        payload = {
            'model': 'claude-sonnet-4-5',
            'max_tokens': 2000,
            'system': SYSTEM_PROMPT,
            'messages': [
                {
                    'role': 'user',
                    'content': f"Generate player positions for this situation: {situation}\n\nIMPORTANT: Return ONLY valid JSON with no markdown formatting, no code blocks, no explanation. Just the raw JSON object."
                }
            ],
            'temperature': 0.7,
        }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return response.json(), response.status_code

    return app
