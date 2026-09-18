This repository contains the `smart_warehouse_bot` implementation, an AI-powered warehouse management system integrating classical AI, uncertainty handling, expert systems, game-playing algorithms, and connectionist models. The system is split into five functional modules and an integration layer.

## Setup Instructions
Before running any module, please ensure your environment is set up properly:
1. Open your terminal or command prompt.
2. Navigate to the project's root directory (`smart_warehouse_bot`).
3. (Optional but recommended) Create and activate a Python virtual environment.
4. Install all necessary dependencies by running: `pip install -r requirements.txt`

## Module A: Planning
The planning module implements both Nilsson's Goal Stack Planning and a Partial Order Planner (POP) to arrange boxes in the warehouse. It features high_level_planner task expansion for pallet loading and a reactive_layer execution layer that repairs plans when environmental disturbances occur. The nonlinear planner is capable of generating parallel execution steps for independent subgoals.

**How to run:**
1. Ensure your terminal is in the project root directory.
2. Run the command: `python part1_planner/plan.py`
3. Check the console output to observe the generated plans and task expansions.

## Module B: Uncertainty
The uncertainty module evaluates sensor readings using six different formalisms: default_logic logic, naive Bayes, certainty factors, Bayesian networks (using pgmpy), Dempster-Shafer theory, and fuzzy_logic logic. It processes noisy telemetry data like vibration, current, and temperature to determine if warehouse equipment is damaged. A comparative analysis tool highlights the disagreements between these methods on edge cases.

**How to run:**
1. Ensure your terminal is in the project root directory.
2. Run the command: `python part2_uncertainty/compare.py`
3. Review the comparative analysis output across all uncertainty formalisms.

## Module C: Game Playing
The game module governs multi-agent competition for shared docking resources using adversarial search. It models the dock negotiation as a zero-sum game, implementing standard Minimax, Alpha-Beta pruning, and Iterative Deepening Search. The algorithms evaluate optimal paths and prune the search space to allow deeper lookahead within time constraints.

**How to run:**
1. Ensure your terminal is in the project root directory.
2. To run the game simulation, execute: `python part3_docking/dock_game.py`
3. To generate search space benchmarks, execute: `python part3_docking/generate_chart.py`

## Module D: Connectionist
The connectionist module features neural network models for pattern recognition and anomaly detection. A Hopfield network built from scratch in NumPy provides content-addressable memory for recalling visual warehouse symbols even when corrupted. Additionally, an RNN built with PyTorch processes temporal sequences of sensor telemetry to predict subtle, drifting anomalies that threshold-based systems miss.

**How to run:**
1. Ensure your terminal is in the project root directory.
2. For an interactive exploration, start Jupyter by running: `jupyter notebook`
3. Open the file `part4_networks/anomaly_notebook.ipynb` and run the cells.
4. Alternatively, you can run the test suite by executing: `pytest part4_networks/tests/test_connectionist.py`

## Module E: Expert System
The expert system provides diagnostic capabilities using a forward-chaining rule engine written in pure Python. It evaluates symptoms flagged by the uncertainty module and infers root causes (e.g., motor failure, structural damage). The system includes an interactive knowledge acquisition component and provides traces explaining its diagnostic reasoning.

**How to run:**
1. Ensure your terminal is in the project root directory.
2. To launch the interactive diagnostic advisor, run: `python part5_expert/expert_advisor.py`
3. To run the interactive knowledge acquisition tool, run: `python part5_expert/add_knowledge.py`

## Integration
The integration layer generates synthetic shift logs encompassing sensor readings, physical disturbances, and dock requests. A central control loop dispatches these events to the appropriate modules, linking Module B's uncertainty evaluations to Module E's diagnostics when damage is suspected. The `main.py` entrypoint orchestrates this entire warehouse simulation workflow.

**How to run:**
1. Ensure your terminal is in the project root directory.
2. Run the command: `python main.py`
3. Follow the console output for the full simulation of the integrated warehouse system.
