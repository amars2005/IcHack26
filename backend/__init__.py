"""
Initialise app, and define GET route
"""
from flask import request, Flask
from flask_cors import CORS
import requests
import json
import os
import base64
import io
import dotenv
import joblib
import lightgbm as lgb
from PIL import Image, ImageDraw, ImageFont
dotenv.load_dotenv()

URL = 'https://api.anthropic.com/v1/messages'
IMAGE_PROMPT = """
You are an expert football/soccer tactical analyst with advanced computer vision capabilities.
Your task is to analyze the provided image of a football pitch and convert player positions into EXACT (x, y) COORDINATES for 22 players.

═══════════════════════════════════════════════════════════════════
1. VISUAL CONTEXT & MAPPING (SPECIFIC TO THIS IMAGE)
═══════════════════════════════════════════════════════════════════
- **Perspective:** High-angle broadcast view.
  - The bottom edge of the image is the "Near Touchline" (y=0).
  - The top edge of the image (near the dugouts) is the "Far Touchline" (y=80).
  - Note: Due to perspective, the far touchline appears visually shorter than the near touchline. You must compensate for this trapezoidal distortion when calculating X coordinates for players further away from the camera.

- **Team Identification:**
  - **ATTACKERS (Red Team):** The team wearing RED kits. They are positioned on the LEFT half of the pitch.
  - **DEFENDERS (Blue Team):** The team wearing BLUE kits. They are positioned on the RIGHT half of the pitch.

- **Ball Location:** Look for the white ball on the ground near the center circle, specifically near the Red player standing closest to the center line.

═══════════════════════════════════════════════════════════════════
2. COORDINATE SYSTEM
═══════════════════════════════════════════════════════════════════
- **Dimensions:** 120m (length) x 80m (width).
- **Origin (0,0):** Bottom-Left Corner of the pitch.
- **Max (120,80):** Top-Right Corner of the pitch.
- **Direction:**
  - Red Team (Attackers) range: Generally x = 0 to x = 60.
  - Blue Team (Defenders) range: Generally x = 60 to x = 120.

═══════════════════════════════════════════════════════════════════
3. CRITICAL OUTPUT RULES (STRICT SCHEMA)
═══════════════════════════════════════════════════════════════════
You must output a JSON object with two arrays: "attackers" and "defenders".

**ATTACKERS ARRAY (Red Team - 11 Objects):**
- MUST contain exactly 11 players.
- IDs: "1" (GK) through "11".
- ID "1" is the Red Goalkeeper (Far Left, near x=0-10).

**DEFENDERS ARRAY (Blue Team - 11 Objects):**
- MUST contain exactly 11 players.
- IDs: "d1" (GK) through "d11".
- ID "d1" is the Blue Goalkeeper (Far Right, near x=110-120).

**BALL ID:**
- Identify the Red player (Attacker) closest to the ball (likely near the center circle) and assign "ball_id" to that player's ID.

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════
Return ONLY valid JSON. No markdown, no code blocks, no explanation.

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
"""

SYSTEM_PROMPT = """You are an expert football/soccer tactical analyst. Your task is to position EXACTLY 22 players(11 attacking, 11 defending) on a football pitch based on the described situation.

═══════════════════════════════════════════════════════════════════
PITCH SPECIFICATIONS
═══════════════════════════════════════════════════════════════════
- Dimensions: 120m(width) × 80m(height)
- Coordinate system: (0, 0) at bottom-left, (120, 80) at top-right
- Attacking team: Positioned on LEFT side, attacking towards RIGHT goal
- Defending team: Positioned on RIGHT side, defending RIGHT goal
- Left goal: x = 0, y = 40 (center of left edge)
- Right goal: x = 120, y = 40 (center of right edge)
- Halfway line: x = 60
- Penalty boxes: 0-18m and 102-120m from each goal line
- Goal areas: 0-6m and 114-120m from each goal line

═══════════════════════════════════════════════════════════════════
TACTICAL KNOWLEDGE
═══════════════════════════════════════════════════════════════════

FORMATIONS & SYSTEMS:
- 4-3-3: Four defenders, three midfielders(often one holding, two box-to-box), three forwards
- 4-4-2: Four defenders, four midfielders(wingers and central pair), two strikers
- 3-5-2: Three center-backs, wing-backs providing width, three central midfielders, two strikers
- 4-2-3-1: Four defenders, two defensive midfielders, three attacking midfielders, one striker
- 3-4-3: Three center-backs, four midfielders, three forwards
- 5-3-2: Five defenders(wing-backs deeper), three midfielders, two forwards

POSITIONAL PRINCIPLES:
- Goalkeeper: Typically 8-15m from goal line(deeper when defending, higher when building up)
- Defensive line: Maintain compactness(8-12m between center-backs, full-backs wider)
- Midfield depth: Stagger positioning to provide passing lanes and cover
- Forward positioning: Create vertical and horizontal space, exploit channels
- Width: Utilize full pitch width(touchlines at y=0 and y=80)
- Compactness: Defensive teams compress space(15-25m between lines)
- Depth: Attacking teams stretch vertically to create space

PHASES OF PLAY:

1. BUILD-UP PLAY:
   - Defenders split wide or drop deep
   - Goalkeeper often higher(x≈18-25m from own goal)
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
   - Defense caught high up pitch(x > 60 for defending team)
   - Numerical advantages for attackers

4. DEFENSIVE ORGANIZATION:
   - Compact defensive block
   - Defenders deeper(x > 85 for defending team)
   - Midfielders protect space in front of defense
   - Forwards press or drop back depending on strategy

5. TRANSITIONS:
   - Defensive transition: Attackers caught high, defenders retreating
   - Attacking transition: Defenders pushing up, quick ball progression

SET PIECES:

Corner Kicks:
- Attackers: Cluster in penalty box(102-120m), near/far post runs, edge of box for clearances
- Ball taker at corner flag(x≈120, y≈0-5 or y≈75-80)
- Defenders: Zonal or man-marking in box, players on posts, keeper on goal line

Free Kicks(Direct):
- Attackers: Wall players, runners, shooters positioned for angles
- Distance from goal determines positioning
- Defenders: Form wall(typically 3-5 players), mark dangerous attackers, keeper positioned

Throw-ins:
- Create numerical advantages near touchline
- Movement to receive or create space
- Opposition positioned to prevent progression

SPATIAL CONCEPTS:
- Half-spaces: Vertical channels between center and wing(y≈20-30 and y≈50-60)
- Central corridor: y≈30-50 (most congested, most direct to goal)
- Wide channels: y≈0-20 and y≈60-80 (space for crosses and overlaps)
- Final third: Last 40m of pitch(x > 80 for attacking team)
- Middle third: Central 40m(x≈40-80)
- Defensive third: First 40m(x < 40 for attacking team)

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

MANDATORY PLAYER IDs(ALL must be included):

ATTACKING TEAM(11 players - IDs are strings):
"1" - Goalkeeper
"2" - Defender(typically right side or right-center)
"3" - Defender(typically center)
"4" - Defender(typically center)
"5" - Defender(typically left side or left-center)
"6" - Midfielder(positioning varies by formation)
"7" - Midfielder(positioning varies by formation)
"8" - Midfielder(positioning varies by formation)
"9" - Forward(positioning varies by formation)
"10" - Forward(often central or playmaker)
"11" - Forward(positioning varies by formation)

DEFENDING TEAM(MUST have EXACTLY 11 players - IDs are strings):
"d1" - Goalkeeper
"d2" - Defender(typically right side or right-center)
"d3" - Defender(typically center)
"d4" - Defender(typically center)
"d5" - Defender(typically left side or left-center)
"d6" - Midfielder(positioning varies by formation)
"d7" - Midfielder(positioning varies by formation)
"d8" - Midfielder(positioning varies by formation)
"d9" - Forward(positioning varies by formation)
"d10" - Forward(often central or playmaker)
"d11" - Forward(positioning varies by formation) ⚠️ CRITICAL: DO NOT FORGET d11!

⚠️ SPECIAL ATTENTION: Player "d11" is FREQUENTLY MISSING. ALWAYS include "d11" in the defenders array.

BEFORE GENERATING - VERIFY THIS CHECKLIST:
1. ✓ Attackers array has EXACTLY 11 objects
2. ✓ Defenders array has EXACTLY 11 objects
3. ✓ Attacker IDs: "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"
4. ✓ Defender IDs: "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11"
5. ✓ SPECIFICALLY verify "11" exists in attackers
6. ✓ SPECIFICALLY verify "d11" exists in defenders(THIS IS CRITICAL!)
7. ✓ Each player has "x", "y", and "id" fields
8. ✓ ball_id matches one attacker ID
9. ✓ No duplicate IDs
10. ✓ Positions within bounds(x: 0-120, y: 0-80)

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════════════

Return ONLY valid JSON with this structure(no markdown, no code blocks, no explanation):

{
  "attackers": [
    {"x": < number > , "y": < number > , "id": "1"},
    {"x": < number > , "y": < number > , "id": "2"},
    {"x": < number > , "y": < number > , "id": "3"},
    {"x": < number > , "y": < number > , "id": "4"},
    {"x": < number > , "y": < number > , "id": "5"},
    {"x": < number > , "y": < number > , "id": "6"},
    {"x": < number > , "y": < number > , "id": "7"},
    {"x": < number > , "y": < number > , "id": "8"},
    {"x": < number > , "y": < number > , "id": "9"},
    {"x": < number > , "y": < number > , "id": "10"},
    {"x": < number > , "y": < number > , "id": "11"}
  ],
  "defenders": [
    {"x": < number > , "y": < number > , "id": "d1"},
    {"x": < number > , "y": < number > , "id": "d2"},
    {"x": < number > , "y": < number > , "id": "d3"},
    {"x": < number > , "y": < number > , "id": "d4"},
    {"x": < number > , "y": < number > , "id": "d5"},
    {"x": < number > , "y": < number > , "id": "d6"},
    {"x": < number > , "y": < number > , "id": "d7"},
    {"x": < number > , "y": < number > , "id": "d8"},
    {"x": < number > , "y": < number > , "id": "d9"},
    {"x": < number > , "y": < number > , "id": "d10"},
    {"x": < number > , "y": < number > , "id": "d11"}
  ],
  "ball_id": "<attacker_id>"
}

⚠️ CRITICAL REMINDER: The defenders array MUST contain all 11 IDs including "d11" at the end.
Ensure coordinates reflect the tactical situation described, formation requirements, and phase of play.
"""


def validate_and_parse_tactical_json(raw_content):
    """
    Cleans, parses, and validates the JSON response from Claude.
    Returns: (valid_data, error_message, status_code)
    """
    try:
        # 1. Remove markdown code blocks if present
        clean_content = raw_content.replace(
            '```json', '').replace('```', '').strip()
        clean_json = json.loads(clean_content)

        # 2. Define Strict Requirements
        required_attacker_ids = [str(i) for i in range(1, 12)]  # "1" to "11"
        required_defender_ids = [
            f"d{i}" for i in range(1, 12)]  # "d1" to "d11"

        attackers = clean_json.get("attackers", [])
        defenders = clean_json.get("defenders", [])

        # 3. Check Counts
        if len(attackers) != 11:
            return None, f"Invalid attacker count: {len(attackers)}. Expected 11.", 400
        if len(defenders) != 11:
            return None, f"Invalid defender count: {len(defenders)}. Expected 11.", 400

        # 4. Check IDs
        current_attacker_ids = {p.get("id") for p in attackers}
        current_defender_ids = {p.get("id") for p in defenders}

        missing_attackers = set(required_attacker_ids) - current_attacker_ids
        missing_defenders = set(required_defender_ids) - current_defender_ids

        if missing_attackers or missing_defenders:
            error_msg = f"Missing IDs - Attackers: {list(missing_attackers)}, Defenders: {list(missing_defenders)}"
            return None, error_msg, 400

        # 5. Check Ball ID
        ball_id = clean_json.get("ball_id")
        if not ball_id or ball_id not in current_attacker_ids:
            return None, f"Invalid ball_id: '{ball_id}'. Must match an existing attacker ID.", 400

        # Success
        return clean_json, None, 200

    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {str(e)}", 500
    except Exception as e:
        return None, f"Unexpected validation error: {str(e)}", 500


def create_annotated_image(base64_image_data, media_type, player_positions, ball_id):
    """
    Create an annotated copy of the image with player positions highlighted.
    Returns base64-encoded annotated image.
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(base64_image_data)
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary (for PNG with transparency)
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')

        draw = ImageDraw.Draw(image)

        # Get image dimensions for scaling
        img_width, img_height = image.size

        # Pitch dimensions (120m x 80m)
        pitch_width = 120
        pitch_height = 80

        # Calculate scale factors (map pitch coordinates to image pixels)
        scale_x = img_width / pitch_width
        scale_y = img_height / pitch_height

        # Circle radius based on image size
        radius = int(min(img_width, img_height) * 0.02)
        font_size = max(12, int(radius * 0.8))

        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()

        attackers = player_positions.get("attackers", [])
        defenders = player_positions.get("defenders", [])

        # Colors (RGB)
        attacker_color = (239, 68, 68)  # Red
        defender_color = (59, 130, 246)  # Blue
        ball_color = (250, 204, 21)  # Yellow
        text_color = (255, 255, 255)  # White

        # Draw attackers
        for player in attackers:
            x = int(player["x"] * scale_x)
            y = int((pitch_height - player["y"]) * scale_y)  # Flip Y axis

            # Check if this player has the ball
            is_ball_carrier = player["id"] == ball_id

            # Draw outer ring for ball carrier
            if is_ball_carrier:
                outer_radius = radius + 6
                draw.ellipse(
                    [(x - outer_radius, y - outer_radius),
                     (x + outer_radius, y + outer_radius)],
                    outline=ball_color,
                    width=4
                )

            # Draw player circle
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=attacker_color,
                outline=(255, 255, 255),
                width=2
            )

            # Draw player ID
            text = str(player["id"])
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text(
                (x - text_width // 2, y - text_height // 2),
                text,
                fill=text_color,
                font=font
            )

        # Draw defenders
        for player in defenders:
            x = int(player["x"] * scale_x)
            y = int((pitch_height - player["y"]) * scale_y)  # Flip Y axis

            # Draw player circle
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=defender_color,
                outline=(255, 255, 255),
                width=2
            )

            # Draw player ID (remove 'd' prefix for cleaner display)
            text = player["id"].replace("d", "")
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text(
                (x - text_width // 2, y - text_height // 2),
                text,
                fill=text_color,
                font=font
            )

        # Add legend
        legend_y = 20
        legend_x = 20
        legend_radius = 10

        # Attacker legend
        draw.ellipse(
            [(legend_x - legend_radius, legend_y - legend_radius),
             (legend_x + legend_radius, legend_y + legend_radius)],
            fill=attacker_color,
            outline=(255, 255, 255),
            width=2
        )
        draw.text((legend_x + legend_radius + 8, legend_y - 8),
                  "Attackers", fill=text_color, font=font)

        # Defender legend
        legend_y += 30
        draw.ellipse(
            [(legend_x - legend_radius, legend_y - legend_radius),
             (legend_x + legend_radius, legend_y + legend_radius)],
            fill=defender_color,
            outline=(255, 255, 255),
            width=2
        )
        draw.text((legend_x + legend_radius + 8, legend_y - 8),
                  "Defenders", fill=text_color, font=font)

        # Ball carrier legend
        legend_y += 30
        draw.ellipse(
            [(legend_x - legend_radius - 3, legend_y - legend_radius - 3),
             (legend_x + legend_radius + 3, legend_y + legend_radius + 3)],
            outline=ball_color,
            width=3
        )
        draw.ellipse(
            [(legend_x - legend_radius, legend_y - legend_radius),
             (legend_x + legend_radius, legend_y + legend_radius)],
            fill=attacker_color,
            outline=(255, 255, 255),
            width=2
        )
        draw.text((legend_x + legend_radius + 8, legend_y - 8),
                  "Ball Carrier", fill=text_color, font=font)

        # Convert back to base64
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=90)
        annotated_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return f"data:image/jpeg;base64,{annotated_base64}"

    except Exception as e:
        print(f"Error creating annotated image: {str(e)}")
        return None


def start_app():
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:5173",
         "http://localhost:5175"], supports_credentials=True)

    @app.route("/", methods=["POST"])
    def predictions():
        if request.method == "POST":
            data = request.get_json()
            print(data)

            if not data:
                return {"error": "No data provided"}, 400

            #  process data
            attackers = data.get("attackers", [])
            defenders = data.get("defenders", [])
            keepers = data.get("keepers", [])
            print(attackers)

            ball_id = data["ball_id"]
            ball_position = next(
                ({"x": p["x"], "y": p["y"]}
                 for p in attackers if p["id"] == ball_id),
                None)

            if ball_position is None:
                raise ValueError(
                    "Ball position could not be determined from ball_id.")

            data_dict = {
                "ball_x": ball_position['x'],
                "ball_y": ball_position['y'],
            }

            for i in range(10):  #  for 11 players
                data_dict[f"p_{i}_x"] = attackers[i]["x"]
                data_dict[f"p_{i}_y"] = attackers[i]["y"]
                data_dict[f'p_{i}_team'] = 1  # attacker

                data_dict[f"p_{i+11}_x"] = defenders[i]["x"]
                data_dict[f"p_{i+11}_y"] = defenders[i]["y"]
                data_dict[f'p_{i+11}_team'] = 0  # defender

            data_dict[f"keeper_1_x"] = keepers[0]["x"]
            data_dict[f"keeper_1_y"] = keepers[0]["y"]
            # keeper for team in entry [0]
            data_dict[f'keeper_1_team'] = 1  # attacker keeper

            data_dict[f"keeper_2_x"] = keepers[1]["x"]
            data_dict[f"keeper_2_y"] = keepers[1]["y"]
            # keeper for team in entry [1]
            data_dict[f'keeper_2_team'] = 0  # defender keeper

            from backend.generalpv.expectedThreatModel import ExpectedThreatModel
            xT = ExpectedThreatModel()
            xT.load_model(os.path.join(os.path.dirname(
                __file__), "models/xT_model.pkl"))

            pxT_value = xT.calculate_expected_threat(**data_dict)

            # random prediction generations
            # import numpy as np
            # xT = np.random.rand(12)
            # p_success = np.random.rand(12)
            # action = np.random.choice(
            #     ["pass", "carry", "shoot"], size=12
            # )

            # evaluations = {action[i]: {
            #     "xT": xT[i], "P(success)": p_success[i]} for i in range(12)}
            return json.dumps(pxT_value)

    @app.route("/generate-positions", methods=["GET"])
    def generate_positions():
        print("Generating positions...")
        if request.method == "GET":
            situation = request.args.get("situation", "")

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

        response = requests.post(URL, headers=headers, json=payload)
        if response.status_code != 200:
            return response.json(), response.status_code

        anthropic_data = response.json()
        raw_content = anthropic_data['content'][0]['text']

        valid_data, error_msg, status_code = validate_and_parse_tactical_json(
            raw_content)

        if valid_data:
            print("Generated positions (validated): Success")
            return valid_data
        else:
            print("Validation error:", error_msg)
            return {"error": error_msg, "raw_response": raw_content}, status_code

    @app.route("/image", methods=['POST'])
    def image_positions():
        if request.method == "POST":
            try:
                req_data = request.get_json()
                raw_input = req_data.get("image", "")

                if "base64," in raw_input:
                    # Split on "base64," and take the data part
                    parts = raw_input.split("base64,", 1)
                    header = parts[0]
                    base64_data = parts[1] if len(parts) > 1 else ""

                    if "image/png" in header:
                        media_type = "image/png"
                    elif "image/jpeg" in header or "image/jpg" in header:
                        media_type = "image/jpeg"
                    elif "image/webp" in header:
                        media_type = "image/webp"
                    elif "image/gif" in header:
                        media_type = "image/gif"
                    else:
                        media_type = "image/jpeg"
                else:
                    base64_data = raw_input
                    media_type = req_data.get("media_type", "image/jpeg")

                # Strip any whitespace from base64 data
                base64_data = base64_data.strip()

                print(
                    f"Sending to Claude -> Type: {media_type} | Length: {len(base64_data)}")

                headers = {
                    'Content-Type': 'application/json',
                    'x-api-key': dotenv.get_key(".env", "API_KEY"),
                    'anthropic-version': '2023-06-01',
                }

                payload = {
                    'model': 'claude-opus-4-5',
                    'max_tokens': 2000,
                    'system': IMAGE_PROMPT,
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": "Analyze this tactical image and generate the player positions JSON."
                                }
                            ]
                        }
                    ],
                    'temperature': 0.5
                }

                response = requests.post(URL, headers=headers, json=payload)

                if response.status_code != 200:
                    error_detail = response.text
                    print(
                        f"Anthropic API Error ({response.status_code}): {error_detail}")
                    try:
                        error_json = response.json()
                        return {"error": f"Anthropic API error: {error_json.get('error', {}).get('message', error_detail)}"}, response.status_code
                    except:
                        return {"error": f"Anthropic API error: {error_detail}"}, response.status_code

                anthropic_data = response.json()
                raw_content = anthropic_data['content'][0]['text']

                valid_data, error_msg, status_code = validate_and_parse_tactical_json(
                    raw_content)

                if valid_data:
                    # Create annotated image with player positions highlighted
                    annotated_image = create_annotated_image(
                        base64_data, media_type, valid_data, valid_data.get(
                            "ball_id")
                    )

                    # Add annotated image to response
                    valid_data["annotated_image"] = annotated_image

                    return valid_data
                else:
                    return {"error": error_msg, "raw_response": raw_content}, status_code

            except Exception as e:
                print(f"Server Error: {str(e)}")
                return {"error": str(e)}, 500

    return app
