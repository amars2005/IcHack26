"""
Initialise app, and define GET route
"""
from flask import request, Flask
from flask_cors import CORS
import requests
import json
import os
import math
import dotenv
import joblib
import lightgbm as lgb
from backend.generalpv.scale_metrics import normalize_value
dotenv.load_dotenv()

# Maximum distance (in meters) from closest opponent for carry to be viable
MAX_CARRY_DIST = 10.0


def get_closest_opponent_distance(ball_position: dict, defenders: list) -> float:
    """
    Calculate the distance to the closest opponent (defender) from the ball position.
    
    Args:
        ball_position: Dict with 'x' and 'y' coordinates of the ball
        defenders: List of defender dicts with 'x' and 'y' coordinates
    
    Returns:
        Distance to the closest opponent in meters
    """
    if not defenders:
        return float('inf')
    
    ball_x = ball_position['x']
    ball_y = ball_position['y']
    
    min_dist = float('inf')
    for defender in defenders:
        dx = defender['x'] - ball_x
        dy = defender['y'] - ball_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < min_dist:
            min_dist = dist
    print("Closest opponent distance:", min_dist)
    return min_dist


MODEL_MODE = "nn"
SYSTEM_PROMPT = """You are an expert football/soccer tactical analyst. Your task is to position EXACTLY 22 players (11 attacking, 11 defending) on a football pitch based on the described situation.

═══════════════════════════════════════════════════════════════════
PITCH SPECIFICATIONS
═══════════════════════════════════════════════════════════════════
- Dimensions: 120m (width) × 80m (height)
- Coordinate system: (0,0) at bottom-left, (120,80) at top-right
- Attacking team: Positioned on LEFT side, attacking towards RIGHT goal
- Defending team: Positioned on RIGHT side, defending RIGHT goal
- Left goal: x=0, y=40 (center of left edge)
- Right goal: x=120, y=40 (center of right edge)
- Halfway line: x=60
- Penalty boxes: 0-18m and 102-120m from each goal line
- Goal areas: 0-6m and 114-120m from each goal line

═══════════════════════════════════════════════════════════════════
TACTICAL KNOWLEDGE
═══════════════════════════════════════════════════════════════════

FORMATIONS & SYSTEMS:
- 4-3-3: Four defenders, three midfielders (often one holding, two box-to-box), three forwards
- 4-4-2: Four defenders, four midfielders (wingers and central pair), two strikers
- 3-5-2: Three center-backs, wing-backs providing width, three central midfielders, two strikers
- 4-2-3-1: Four defenders, two defensive midfielders, three attacking midfielders, one striker
- 3-4-3: Three center-backs, four midfielders, three forwards
- 5-3-2: Five defenders (wing-backs deeper), three midfielders, two forwards

POSITIONAL PRINCIPLES:
- Goalkeeper: Typically 8-15m from goal line (deeper when defending, higher when building up)
- Defensive line: Maintain compactness (8-12m between center-backs, full-backs wider)
- Midfield depth: Stagger positioning to provide passing lanes and cover
- Forward positioning: Create vertical and horizontal space, exploit channels
- Width: Utilize full pitch width (touchlines at y=0 and y=80)
- Compactness: Defensive teams compress space (15-25m between lines)
- Depth: Attacking teams stretch vertically to create space

PHASES OF PLAY:

1. BUILD-UP PLAY:
   - Defenders split wide or drop deep
   - Goalkeeper often higher (x≈18-25m from own goal)
   - Midfielders drop to receive
   - Forwards pin opposition defenders

2. ATTACKING PHASE:
   - Forwards high and wide
   - Midfielders push up to support
   - Full-backs advance or overlap
   - Create numerical advantages in wide areas or centrally

3. COUNTER-ATTACK:
   - Quick vertical progression
   - Forwards sprint into space
   - Defense caught high up pitch (x > 60 for defending team)
   - Numerical advantages for attackers

4. DEFENSIVE ORGANIZATION:
   - Compact defensive block
   - Defenders deeper (x > 85 for defending team)
   - Midfielders protect space in front of defense
   - Forwards press or drop back depending on strategy

5. TRANSITIONS:
   - Defensive transition: Attackers caught high, defenders retreating
   - Attacking transition: Defenders pushing up, quick ball progression

SET PIECES:

Corner Kicks:
- Attackers: Cluster in penalty box (102-120m), near/far post runs, edge of box for clearances
- Ball taker at corner flag (x≈120, y≈0-5 or y≈75-80)
- Defenders: Zonal or man-marking in box, players on posts, keeper on goal line

Free Kicks (Direct):
- Attackers: Wall players, runners, shooters positioned for angles
- Distance from goal determines positioning
- Defenders: Form wall (typically 3-5 players), mark dangerous attackers, keeper positioned

Throw-ins:
- Create numerical advantages near touchline
- Movement to receive or create space
- Opposition positioned to prevent progression

SPATIAL CONCEPTS:
- Half-spaces: Vertical channels between center and wing (y≈20-30 and y≈50-60)
- Central corridor: y≈30-50 (most congested, most direct to goal)
- Wide channels: y≈0-20 and y≈60-80 (space for crosses and overlaps)
- Final third: Last 40m of pitch (x > 80 for attacking team)
- Middle third: Central 40m (x≈40-80)
- Defensive third: First 40m (x < 40 for attacking team)

PLAYER-SPECIFIC POSITIONING:
- Goalkeeper: Mobile, command penalty area, starting point for build-up
- Center-backs: Vertical positioning based on defensive line, horizontal spacing for coverage
- Full-backs: Balance defensive duties with attacking support, provide width
- Holding midfielder: Screen defense, positioned between defense and midfield lines
- Box-to-box midfielder: Dynamic positioning, support attack and defense
- Attacking midfielder: Between midfield and forward lines, creative spaces
- Wingers: Width, 1v1 situations, cutting inside or staying wide
- Striker: Pin center-backs, create space, finish chances

═══════════════════════════════════════════════════════════════════
CRITICAL REQUIREMENTS
═══════════════════════════════════════════════════════════════════

MANDATORY PLAYER IDs (ALL must be included):

ATTACKING TEAM (11 players - IDs are strings):
"1"  - Goalkeeper
"2"  - Defender (typically right side or right-center)
"3"  - Defender (typically center)
"4"  - Defender (typically center)
"5"  - Defender (typically left side or left-center)
"6"  - Midfielder (positioning varies by formation)
"7"  - Midfielder (positioning varies by formation)
"8"  - Midfielder (positioning varies by formation)
"9"  - Forward (positioning varies by formation)
"10" - Forward (often central or playmaker)
"11" - Forward (positioning varies by formation)

DEFENDING TEAM (MUST have EXACTLY 11 players - IDs are strings):
"d1"  - Goalkeeper
"d2"  - Defender (typically right side or right-center)
"d3"  - Defender (typically center)
"d4"  - Defender (typically center)
"d5"  - Defender (typically left side or left-center)
"d6"  - Midfielder (positioning varies by formation)
"d7"  - Midfielder (positioning varies by formation)
"d8"  - Midfielder (positioning varies by formation)
"d9"  - Forward (positioning varies by formation)
"d10" - Forward (often central or playmaker)
"d11" - Forward (positioning varies by formation) ⚠️ CRITICAL: DO NOT FORGET d11!

⚠️ SPECIAL ATTENTION: Player "d11" is FREQUENTLY MISSING. ALWAYS include "d11" in the defenders array.

BEFORE GENERATING - VERIFY THIS CHECKLIST:
1. ✓ Attackers array has EXACTLY 11 objects
2. ✓ Defenders array has EXACTLY 11 objects
3. ✓ Attacker IDs: "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"
4. ✓ Defender IDs: "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11"
5. ✓ SPECIFICALLY verify "11" exists in attackers
6. ✓ SPECIFICALLY verify "d11" exists in defenders (THIS IS CRITICAL!)
7. ✓ Each player has "x", "y", and "id" fields
8. ✓ ball_id matches one attacker ID
9. ✓ No duplicate IDs
10. ✓ Positions within bounds (x: 0-120, y: 0-80)

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════

Return ONLY valid JSON with this structure (no markdown, no code blocks, no explanation):

{
  "attackers": [
    {"x": <number>, "y": <number>, "id": "1"},
    {"x": <number>, "y": <number>, "id": "2"},
    {"x": <number>, "y": <number>, "id": "3"},
    {"x": <number>, "y": <number>, "id": "4"},
    {"x": <number>, "y": <number>, "id": "5"},
    {"x": <number>, "y": <number>, "id": "6"},
    {"x": <number>, "y": <number>, "id": "7"},
    {"x": <number>, "y": <number>, "id": "8"},
    {"x": <number>, "y": <number>, "id": "9"},
    {"x": <number>, "y": <number>, "id": "10"},
    {"x": <number>, "y": <number>, "id": "11"}
  ],
  "defenders": [
    {"x": <number>, "y": <number>, "id": "d1"},
    {"x": <number>, "y": <number>, "id": "d2"},
    {"x": <number>, "y": <number>, "id": "d3"},
    {"x": <number>, "y": <number>, "id": "d4"},
    {"x": <number>, "y": <number>, "id": "d5"},
    {"x": <number>, "y": <number>, "id": "d6"},
    {"x": <number>, "y": <number>, "id": "d7"},
    {"x": <number>, "y": <number>, "id": "d8"},
    {"x": <number>, "y": <number>, "id": "d9"},
    {"x": <number>, "y": <number>, "id": "d10"},
    {"x": <number>, "y": <number>, "id": "d11"}
  ],
  "ball_id": "<attacker_id>"
}

⚠️ CRITICAL REMINDER: The defenders array MUST contain all 11 IDs including "d11" at the end.
Ensure coordinates reflect the tactical situation described, formation requirements, and phase of play.
"""


def start_app():
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:5173",
         "http://localhost:5174", "http://localhost:5175"], supports_credentials=True)

    @app.route("/test", methods=["POST"])
    def test():
        """
        Test route to verify backend is operational.
        """
        try:
            if request.method == "POST":
                data = request.get_json()

                if not data:
                    return {"error": "No data provided"}, 400

                #  process data
                attackers = data.get("attackers", [])
                defenders = data.get("defenders", [])
                keepers = data.get("keepers", [])

                ball_id = data["ball_id"]
                # Ball carrier could be in attackers or keepers
                ball_position = next(
                    ({"x": p["x"], "y": p["y"]}
                     for p in attackers if p["id"] == ball_id),
                    None)

                # If not in attackers, check keepers
                if ball_position is None:
                    ball_position = next(
                        ({"x": k["x"], "y": k["y"]}
                         for k in keepers if k["id"] == ball_id),
                        None)

                if ball_position is None:
                    return {"error": f"Ball position could not be determined from ball_id: {ball_id}"}, 400

                data_dict = {
                    "ball_x": ball_position['x'],
                    "ball_y": ball_position['y'],
                }

                # Attackers: p0 through p9 (10 outfield players)
                for i in range(len(attackers)):
                    data_dict[f"p{i}_x"] = attackers[i]["x"]
                    data_dict[f"p{i}_y"] = attackers[i]["y"]
                    data_dict[f'p{i}_team'] = 1  # attacker

                # Defenders: p10 through p19 (10 outfield players)
                for i in range(len(defenders)):
                    data_dict[f"p{i+10}_x"] = defenders[i]["x"]
                    data_dict[f"p{i+10}_y"] = defenders[i]["y"]
                    data_dict[f'p{i+10}_team'] = 0  # defender

                data_dict[f"keeper_1_x"] = keepers[0]["x"]
                data_dict[f"keeper_1_y"] = keepers[0]["y"]
                # keeper for team in entry [0]
                data_dict[f'keeper_1_team'] = 1  # attacker keeper

                data_dict[f"keeper_2_x"] = keepers[1]["x"]
                data_dict[f"keeper_2_y"] = keepers[1]["y"]
                # keeper for team in entry [1]
                data_dict[f'keeper_2_team'] = 0  # defender keeper

                try:
                    if MODEL_MODE == "nn":
                        from backend.generalpv.expectedThreatModelNN import ExpectedThreatModelNN
                        xT = ExpectedThreatModelNN()
                        xT.load_model()
                    else:
                        from backend.generalpv.expectedThreatModel import ExpectedThreatModel
                        xT = ExpectedThreatModel(skip_training=True)
                        xT.load_model(os.path.join(os.path.dirname(
                            __file__), "models/xt_nn_model.pkl"))

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return {"error": f"Model loading failed: {e}"}, 500

                pxT_value = xT.calculate_expected_threat(**data_dict)

                # Generate heatmap if supported (NN model only)
                heatmap_data = None
                if MODEL_MODE == "nn" and hasattr(xT, 'generate_heatmap'):
                    try:
                        # Use 48x32 grid for higher resolution
                        heatmap_data = xT.generate_heatmap(
                            grid_size=(48, 32), **data_dict)
                    except Exception as e:
                        print(f"Heatmap generation failed: {e}")
                        heatmap_data = None

                print("Predicted xT:", pxT_value)

                response_data = {
                    "xT": pxT_value,
                    "heatmap": heatmap_data
                }
                return json.dumps(response_data)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Server error: {str(e)}"}, 500

    @app.route("/", methods=["POST"])
    def predictions():
        """
        Generate predictions for best actions given all player positions.
        """
        if request.method == "POST":
            data = request.get_json()

            if not data:
                return {"error": "No data provided"}, 400

            #  process data
            attackers = data.get("attackers", [])
            defenders = data.get("defenders", [])
            keepers = data.get("keepers", [])

            ball_id = data["ball_id"]
            # Ball carrier could be in attackers or keepers
            ball_position = next(
                ({"x": p["x"], "y": p["y"]}
                 for p in attackers if p["id"] == ball_id),
                None)

            # If not in attackers, check keepers
            if ball_position is None:
                ball_position = next(
                    ({"x": k["x"], "y": k["y"]}
                     for k in keepers if k["id"] == ball_id),
                    None)

            if ball_position is None:
                return {"error": f"Ball position could not be determined from ball_id: {ball_id}"}, 400

            data_dict = {
                "ball_x": ball_position['x'],
                "ball_y": ball_position['y'],
            }

            # Attackers: p0 through p9 (10 outfield players)
            for i in range(len(attackers)):
                data_dict[f"p{i}_x"] = attackers[i]["x"]
                data_dict[f"p{i}_y"] = attackers[i]["y"]
                data_dict[f'p{i}_team'] = 1  # attacker

            # Defenders: p10 through p19 (10 outfield players)
            for i in range(len(defenders)):
                data_dict[f"p{i+10}_x"] = defenders[i]["x"]
                data_dict[f"p{i+10}_y"] = defenders[i]["y"]
                data_dict[f'p{i+10}_team'] = 0  # defender

            data_dict["keeper1_x"] = keepers[0]["x"]
            data_dict["keeper1_y"] = keepers[0]["y"]
            data_dict['keeper1_team'] = 1  # attacker keeper

            data_dict["keeper2_x"] = keepers[1]["x"]
            data_dict["keeper2_y"] = keepers[1]["y"]
            data_dict['keeper2_team'] = 0  # defender keeper

            # ---------------- data processing done for required dict format: data_dict----------------#

            actions = {}

            # SHOOT MODEL
            from backend.generalpv.xg import ExpectedGoalModel
            xg_model = ExpectedGoalModel(skip_training=True)
            model_path = os.path.join(os.path.dirname(os.path.dirname(
                __file__)), "models/xg_model_360.pkl")
            xg_model.load_model(model_path)
            xg_value = xg_model.calculate_expected_goal(**data_dict)
            actions["shoot"] = {"xG": normalize_value("xG", xg_value)}

            # CARRY MODEL
            from backend.generalpv.carryModel import CarryModel
            carry_model = CarryModel()
            
            # Check if closest opponent is within MIN_CARRY_DIST - if so, carry is not viable
            closest_opponent_dist = get_closest_opponent_distance(ball_position, defenders)
            
            if closest_opponent_dist > MAX_CARRY_DIST:
                # Opponent too close - carry not viable
                actions["carry"] = {"xT": None}
            elif carry_model.is_trained:
                carry_result = carry_model.calculate_carry_score(data_dict)
                if carry_result:
                    actions["carry"] = {
                        "xT": normalize_value("xT", carry_result["predicted_xt"]),
                        "xT_gain": carry_result["xt_gain"],
                        "score": carry_result["score"]
                    }
                else:
                    actions["carry"] = {"xT": None}
            else:
                actions["carry"] = {"xT": None}

            # PASS MODEL - Using PassScoreModel for comprehensive evaluation
            from backend.generalpv.passScoreModel import PassScoreModel
            pass_score_model = PassScoreModel()
            pass_score_model.load_models()

            # Get current xT for reference
            current_xT = pass_score_model.get_current_xT(
                ball_position, attackers, defenders, keepers
            )
            actions["current_xT"] = normalize_value("xT", current_xT)

            # Calculate pass scores for all targets
            pass_scores = pass_score_model.calculate_pass_scores(
                ball_position=ball_position,
                attackers=attackers,
                defenders=defenders,
                keepers=keepers,
                ball_id=ball_id,
                data_dict=data_dict
            )

            # Format pass results for frontend
            for player_id, metrics in pass_scores.items():
                actions[f"pass_to_{player_id}"] = {
                    "xT": normalize_value("target_xT", metrics["target_xT"]),
                    "P(success)": normalize_value("success_prob", metrics["success_prob"]),
                    "reward": normalize_value("reward", metrics["reward"]),
                    "risk": normalize_value("risk", metrics["risk"]),
                    "score": normalize_value("score", metrics["score"]),
                    "opponent_xT": normalize_value("opponent_xT", metrics["opponent_xT"]),
                    "interception_point": metrics["interception_point"]
                }

            print("Predicted actions:", actions)

            return json.dumps(actions)

    @app.route("/generate-positions", methods=["GET"])
    def generate_positions():
        """
        Call Claude to generate player positions based on input situation
        """
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
            anthropic_data = response.json()
            raw_content = anthropic_data['content'][0]['text']

            try:
                # Remove markdown code blocks if present
                clean_content = raw_content.replace(
                    '```json', '').replace('```', '').strip()
                clean_json = json.loads(clean_content)

                # VALIDATION: Ensure all 22 players are present
                required_attacker_ids = [
                    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
                required_defender_ids = [
                    "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11"]

                attackers = clean_json.get("attackers", [])
                defenders = clean_json.get("defenders", [])

                attacker_ids = [p["id"] for p in attackers]
                defender_ids = [p["id"] for p in defenders]

                # Check if all required IDs are present
                missing_attackers = [
                    id for id in required_attacker_ids if id not in attacker_ids]
                missing_defenders = [
                    id for id in required_defender_ids if id not in defender_ids]

                if missing_attackers or missing_defenders:
                    error_msg = f"Missing players - Attackers: {missing_attackers}, Defenders: {missing_defenders}"
                    print("Validation error:", error_msg)
                    return {"error": error_msg, "partial_data": clean_json}, 400

                if len(attackers) != 11 or len(defenders) != 11:
                    error_msg = f"Invalid player count - Attackers: {len(attackers)}, Defenders: {len(defenders)}"
                    print("Validation error:", error_msg)
                    return {"error": error_msg, "partial_data": clean_json}, 400

                print("Generated positions (validated):", clean_json)
                return clean_json

            except json.JSONDecodeError as e:
                print("JSON parse error:", str(e))
                return {"error": "Failed to parse model output", "raw": raw_content}, 500
        else:
            return response.json(), response.status_code

    return app
