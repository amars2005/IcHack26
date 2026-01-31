"""
Initialise app, and define GET route
"""
from flask import request, Flask
import json
import pandas as pd


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
            xT = np.rand(12)
            p_success = np.rand(12)
            action = np.random.choice(
                ["pass", "carry", "shoot"], size=12
            )

            evaluations = {action[i]: {
                "xT": xT[i], "P(success)": p_success[i]} for i in range(len(data))}
            return json.dumps(evaluations, indent=4)

    return app
