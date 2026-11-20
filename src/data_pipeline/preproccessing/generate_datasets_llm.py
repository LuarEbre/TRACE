import json
import random
import sys
import math
from pathlib import Path

# pedestrian = true? flag for spawning manually 
# llm_output.update() -> append render configs
# data always appended afterwards (static): 
""" "horizon": 1000,"crash_vehicle_done": false,
    "crash_object_done": false,
    "out_of_road_done": false,
    "use_render": true,
    "manual_control": false,
    "show_interface": true,
    "daytime": "22:00",
    "show_sidewalk": true,
    "show_crosswalk": true,
    "random_traffic": false,
    "num_scenarios" = 1,
"""

# prompt for llm model 
SYSTEM_PROMPT = (
    "You are a MetaDrive Generator. Translate the user description into a valid JSON configuration."
    "Output ONLY raw JSON without markdown formatting or explanations. "
    "You will either use "
    "1) 'type': 'block_sequence' with 'config' as a string of block codes (S=Straight, C=Curve, O=Roundabout, T=T-Intersection, r=In-Ramp, R=Out-Ramp) when specific road elements are requested, OR "
    "2) 'type': 'block_num' with 'config' as an integer (e.g. 5) when a random or unspecified road layout is requested."
    "Default start_seed is null unless specified otherwise."
)

# python list with dictionary elements as templates
# (random)lane_width, (random)lane_num, exit_length, type (block_seq or num), config (string/int)
ROAD_TEMPLATES = [
    {"type": "block_sequence", "config": "SSSSS", "lane_num": 4, "random_lane_num": False, "lane_width": 3.75, "random_lane_width": False, "exit_length": 60},
    {"type": "block_sequence", "config": "SCSCS", "lane_num": 3, "random_lane_num": False, "lane_width": 3.75, "random_lane_width": False, "exit_length": 60},
    {"type": "block_sequence", "config": "SrSSR", "lane_num": 3, "random_lane_num": False, "lane_width": 3.75, "random_lane_width": False, "exit_length": 60},
    {"type": "block_sequence", "config": "SCCCS", "lane_num": 2, "random_lane_num": False, "lane_width": 3.5,  "random_lane_width": False, "exit_length": 50},
    {"type": "block_sequence", "config": "CCCCC", "lane_num": 2, "random_lane_num": False, "lane_width": 3.0,  "random_lane_width": False, "exit_length": 40},
    {"type": "block_sequence", "config": "SCOCS", "lane_num": 2, "random_lane_num": False, "lane_width": 3.5,  "random_lane_width": False, "exit_length": 50},
    {"type": "block_sequence", "config": "SOSOS", "lane_num": 2, "random_lane_num": False, "lane_width": 3.25, "random_lane_width": False, "exit_length": 40},
    {"type": "block_sequence", "config": "STCTS", "lane_num": 2, "random_lane_num": False, "lane_width": 3.2,  "random_lane_width": False, "exit_length": 40},
    {"type": "block_sequence", "config": "SCOTS", "lane_num": 2, "random_lane_num": False, "lane_width": 3.5,  "random_lane_width": False, "exit_length": 50},
    {"type": "block_num",      "config": 3,       "lane_num": 2, "random_lane_num": True,  "lane_width": 3.5,  "random_lane_width": True,  "exit_length": 50},
    {"type": "block_num",      "config": 5,       "lane_num": 2, "random_lane_num": True,  "random_lane_width": True,  "lane_width": 3.5,  "exit_length": 50},
    {"type": "block_num",      "config": 8,       "lane_num": 3, "random_lane_num": False, "lane_width": 3.75, "random_lane_width": False, "exit_length": 50},
    # short roads

    {"type": "block_sequence", "config": "ST",        "lane_num": 2, "random_lane_num": True, "lane_width": 3.1, "random_lane_width": False, "exit_length": 60},
    {"type": "block_sequence", "config": "SS",        "lane_num": 2, "random_lane_num": False, "lane_width": 3.5,  "random_lane_width": False, "exit_length": 40},
    {"type": "block_sequence", "config": "SO",        "lane_num": 2, "random_lane_num": False, "lane_width": 3.25, "random_lane_width": False, "exit_length": 35},
    {"type": "block_sequence", "config": "SCT",       "lane_num": 1, "random_lane_num": False, "lane_width": 2.8,  "random_lane_width": False, "exit_length": 35},
    {"type": "block_sequence", "config": "rSR",       "lane_num": 4, "random_lane_num": False, "lane_width": 4.0,  "random_lane_width": False, "exit_length": 55},
    {"type": "block_num",      "config": 2,           "lane_num": 2, "random_lane_num": True,  "lane_width": 3.5,  "random_lane_width": True,  "exit_length": 40},
    {"type": "block_num",      "config": 1,           "lane_num": 1, "random_lane_num": False,  "lane_width": 2.5,  "random_lane_width": True,  "exit_length": 40},

    # Low width
    {"type": "block_sequence", "config": "SCCS",      "lane_num": 1, "random_lane_num": False, "lane_width": 2.6,  "random_lane_width": False, "exit_length": 35},
    {"type": "block_sequence", "config": "CCCCCC",    "lane_num": 1, "random_lane_num": False, "lane_width": 2.75, "random_lane_width": False, "exit_length": 30},
    {"type": "block_sequence", "config": "STCTCS",    "lane_num": 2, "random_lane_num": False, "lane_width": 2.8,  "random_lane_width": False, "exit_length": 35},
    {"type": "block_sequence", "config": "SOCOS",     "lane_num": 2, "random_lane_num": False, "lane_width": 2.9,  "random_lane_width": False, "exit_length": 40},
    {"type": "block_num",      "config": 4,           "lane_num": 1, "random_lane_num": False, "lane_width": 2.7,  "random_lane_width": False, "exit_length": 35},

    # super low width
    {"type": "block_sequence", "config": "STCTS",     "lane_num": 2, "random_lane_num": False, "lane_width": 1.5,  "random_lane_width": False, "exit_length": 35},
    {"type": "block_sequence", "config": "OTCC",      "lane_num": 1, "random_lane_num": False, "lane_width": 1.6,  "random_lane_width": False, "exit_length": 35},
    {"type": "block_sequence", "config": "SCOTS",     "lane_num": 2, "random_lane_num": False, "lane_width": 1.8,  "random_lane_width": False, "exit_length": 40},
    {"type": "block_sequence", "config": "rSSR",      "lane_num": 2, "random_lane_num": True,  "lane_width": 2.0,  "random_lane_width": False, "exit_length": 40},

    {"type": "block_num",      "config": 3,           "lane_num": 1, "random_lane_num": False, "lane_width": 1.1,  "random_lane_width": False, "exit_length": 30},
    {"type": "block_num",      "config": 5,           "lane_num": 1, "random_lane_num": False, "lane_width": 1.4,  "random_lane_width": True,  "exit_length": 35},

    # High width
    {"type": "block_sequence", "config": "SSSSSS",    "lane_num": 5, "random_lane_num": False, "lane_width": 4.2,  "random_lane_width": False, "exit_length": 75},
    {"type": "block_sequence", "config": "SrSSSR",    "lane_num": 6, "random_lane_num": False, "lane_width": 4.5,  "random_lane_width": False, "exit_length": 80},
    {"type": "block_sequence", "config": "SCSCSCS",   "lane_num": 5, "random_lane_num": False, "lane_width": 4.0,  "random_lane_width": False, "exit_length": 70},
    {"type": "block_sequence", "config": "rRSRSR",    "lane_num": 4, "random_lane_num": False, "lane_width": 4.25, "random_lane_width": False, "exit_length": 65},
    {"type": "block_num",      "config": 6,           "lane_num": 5, "random_lane_num": False, "lane_width": 4.1,  "random_lane_width": False, "exit_length": 70},

    # Long street
    {"type": "block_sequence", "config": "SCOTSCRS",  "lane_num": 3, "random_lane_num": False, "lane_width": 3.6,  "random_lane_width": False, "exit_length": 60},
    {"type": "block_sequence", "config": "CCSOSTCCS", "lane_num": 2, "random_lane_num": False, "lane_width": 3.1,  "random_lane_width": False, "exit_length": 45},
    {"type": "block_sequence", "config": "SSSSCSSSSS","lane_num": 4, "random_lane_num": False, "lane_width": 3.85, "random_lane_width": False, "exit_length": 70},
    {"type": "block_num",      "config": 10,          "lane_num": 3, "random_lane_num": True,  "lane_width": 3.5,  "random_lane_width": True,  "exit_length": 55},
    {"type": "block_num",      "config": 14,          "lane_num": 2, "random_lane_num": True,  "lane_width": 3.5,  "random_lane_width": True,  "exit_length": 60},

    {"type": "block_sequence", "config": "OTCC",      "lane_num": 2, "random_lane_num": False, "lane_width": 3.3,  "random_lane_width": False, "exit_length": 45},
    {"type": "block_sequence", "config": "OSCSO",     "lane_num": 2, "random_lane_num": False, "lane_width": 3.4,  "random_lane_width": False, "exit_length": 45},
    {"type": "block_sequence", "config": "TSCTS",     "lane_num": 2, "random_lane_num": False, "lane_width": 3.25, "random_lane_width": False, "exit_length": 40},
    {"type": "block_sequence", "config": "TORT",      "lane_num": 2, "random_lane_num": False, "lane_width": 3.2,  "random_lane_width": False, "exit_length": 40},
    {"type": "block_sequence", "config": "CSOS",      "lane_num": 2, "random_lane_num": False, "lane_width": 3.5,  "random_lane_width": False, "exit_length": 50},
    {"type": "block_sequence", "config": "CTSCR",     "lane_num": 3, "random_lane_num": False, "lane_width": 3.6,  "random_lane_width": False, "exit_length": 55},
    {"type": "block_sequence", "config": "rSSRS",     "lane_num": 4, "random_lane_num": False, "lane_width": 3.8,  "random_lane_width": False, "exit_length": 65},
    {"type": "block_sequence", "config": "RSCSr",     "lane_num": 3, "random_lane_num": False, "lane_width": 3.75, "random_lane_width": False, "exit_length": 60},
    {"type": "block_sequence", "config": "OCrSRT",    "lane_num": 3, "random_lane_num": False, "lane_width": 3.5,  "random_lane_width": False, "exit_length": 50},
    {"type": "block_sequence", "config": "COrT",      "lane_num": 2, "random_lane_num": False, "lane_width": 3.1,  "random_lane_width": False, "exit_length": 40},

    {"type": "block_sequence", "config": "SCSCS",     "lane_num": 3, "random_lane_num": False, "lane_width": 3.5,  "random_lane_width": True,  "exit_length": 50},
    {"type": "block_sequence", "config": "SOCTS",     "lane_num": 2, "random_lane_num": True,  "lane_width": 3.5,  "random_lane_width": False, "exit_length": 45},
    {"type": "block_sequence", "config": "rSSSR",     "lane_num": 4, "random_lane_num": False, "lane_width": 3.75, "random_lane_width": True,  "exit_length": 60},
    {"type": "block_sequence", "config": "CCC",       "lane_num": 2, "random_lane_num": True,  "lane_width": 2.9,  "random_lane_width": False, "exit_length": 35},
    {"type": "block_sequence", "config": "TSOT",      "lane_num": 2, "random_lane_num": False, "lane_width": 3.25, "random_lane_width": True,  "exit_length": 40},
    {"type": "block_num",      "config": 4,           "lane_num": 3, "random_lane_num": False, "lane_width": 3.5,  "random_lane_width": True,  "exit_length": 50},
    {"type": "block_num",      "config": 6,           "lane_num": 2, "random_lane_num": True,  "lane_width": 3.2,  "random_lane_width": False, "exit_length": 45},
    {"type": "block_num",      "config": 7,           "lane_num": 4, "random_lane_num": False, "lane_width": 4.0,  "random_lane_width": True,  "exit_length": 65},
    {"type": "block_num",      "config": 9,           "lane_num": 1, "random_lane_num": True,  "lane_width": 2.8,  "random_lane_width": False, "exit_length": 40},
    {"type": "block_num",      "config": 11,          "lane_num": 3, "random_lane_num": False, "lane_width": 3.6,  "random_lane_width": True,  "exit_length": 55},
]

# need_inverse_traffic, traffic_density, traffic_mode, accident_prob, static_traffic_object
DRIVING_BEHAVIOUR = [
    {"need_inverse_traffic": False, "traffic_density": 0.0,  "traffic_mode": "respawn", "accident_prob": 0.0,  "static_traffic_object": False},
    {"need_inverse_traffic": False, "traffic_density": 0.0,  "traffic_mode": "respawn", "accident_prob": 0.25, "static_traffic_object": True},
    {"need_inverse_traffic": False, "traffic_density": 0.0,  "traffic_mode": "respawn", "accident_prob": 0.55, "static_traffic_object": True},
    {"need_inverse_traffic": False, "traffic_density": 0.0,  "traffic_mode": "respawn", "accident_prob": 0.85, "static_traffic_object": True},

    {"need_inverse_traffic": False, "traffic_density": 0.08, "traffic_mode": "respawn", "accident_prob": 0.0,  "static_traffic_object": False},
    {"need_inverse_traffic": True,  "traffic_density": 0.1,  "traffic_mode": "respawn", "accident_prob": 0.0,  "static_traffic_object": False},
    {"need_inverse_traffic": False, "traffic_density": 0.12, "traffic_mode": "respawn", "accident_prob": 0.15, "static_traffic_object": True},
    {"need_inverse_traffic": True,  "traffic_density": 0.15, "traffic_mode": "respawn", "accident_prob": 0.3,  "static_traffic_object": True},
    {"need_inverse_traffic": False, "traffic_density": 0.15, "traffic_mode": "trigger", "accident_prob": 0.2,  "static_traffic_object": True},

    {"need_inverse_traffic": False, "traffic_density": 0.2,  "traffic_mode": "respawn", "accident_prob": 0.0,  "static_traffic_object": False},
    {"need_inverse_traffic": True,  "traffic_density": 0.25, "traffic_mode": "respawn", "accident_prob": 0.0,  "static_traffic_object": False},
    {"need_inverse_traffic": True,  "traffic_density": 0.25, "traffic_mode": "respawn", "accident_prob": 0.2,  "static_traffic_object": True},
    {"need_inverse_traffic": False, "traffic_density": 0.3,  "traffic_mode": "respawn", "accident_prob": 0.45, "static_traffic_object": True},
    {"need_inverse_traffic": True,  "traffic_density": 0.35, "traffic_mode": "respawn", "accident_prob": 0.65, "static_traffic_object": True},
    {"need_inverse_traffic": True,  "traffic_density": 0.3,  "traffic_mode": "trigger", "accident_prob": 0.0,  "static_traffic_object": False},

    {"need_inverse_traffic": False, "traffic_density": 0.4,  "traffic_mode": "respawn", "accident_prob": 0.0,  "static_traffic_object": False},
    {"need_inverse_traffic": True,  "traffic_density": 0.4,  "traffic_mode": "respawn", "accident_prob": 0.3,  "static_traffic_object": True},
    {"need_inverse_traffic": True,  "traffic_density": 0.4,  "traffic_mode": "respawn", "accident_prob": 0.7,  "static_traffic_object": True},
    {"need_inverse_traffic": False, "traffic_density": 0.45, "traffic_mode": "respawn", "accident_prob": 0.5,  "static_traffic_object": True},
    {"need_inverse_traffic": True,  "traffic_density": 0.5,  "traffic_mode": "respawn", "accident_prob": 0.25, "static_traffic_object": True},
    {"need_inverse_traffic": False, "traffic_density": 0.5,  "traffic_mode": "trigger", "accident_prob": 0.6,  "static_traffic_object": True},

    {"need_inverse_traffic": False, "traffic_density": 0.6,  "traffic_mode": "respawn", "accident_prob": 0.1,  "static_traffic_object": True},
    {"need_inverse_traffic": True,  "traffic_density": 0.65, "traffic_mode": "respawn", "accident_prob": 0.4,  "static_traffic_object": True},
    {"need_inverse_traffic": False, "traffic_density": 0.7,  "traffic_mode": "respawn", "accident_prob": 0.8,  "static_traffic_object": True},
    {"need_inverse_traffic": True,  "traffic_density": 0.75, "traffic_mode": "respawn", "accident_prob": 0.9,  "static_traffic_object": True},
    {"need_inverse_traffic": False, "traffic_density": 0.8,  "traffic_mode": "respawn", "accident_prob": 0.0,  "static_traffic_object": False},
    {"need_inverse_traffic": True,  "traffic_density": 0.85, "traffic_mode": "trigger", "accident_prob": 0.75, "static_traffic_object": True},

    {"need_inverse_traffic": False, "traffic_density": 0.25, "traffic_mode": "trigger", "accident_prob": 0.5,  "static_traffic_object": True},
    {"need_inverse_traffic": True,  "traffic_density": 0.45, "traffic_mode": "trigger", "accident_prob": 0.9,  "static_traffic_object": True},
]

# start_seed, num_scenarios
INITIALIZATION = []

def log_uniform_int(low: int, high: int) -> int:
    return int(round(math.exp(random.uniform(math.log(low), math.log(high)))))

def log_uniform_float(low: float, high: float, decimals: int = 2) -> float:
    val = math.exp(random.uniform(math.log(low), math.log(high)))
    return round(val, decimals)

def generate_random_datasets(amount:int) -> list[dict]:
    data_sets = []
    # generate random variance of correct JSON datasets 
    for _ in range(amount):

        road = random.choice(ROAD_TEMPLATES).copy()
        behaviour = random.choice(DRIVING_BEHAVIOUR).copy()
        start_seed = None if random.random() < 0.75 else random.randint(1, 10000) # 75% of cases the seed is None
        #num_scenarios = 1 # force to 1?  

        # logical adjustments
        # if lane_number > 3 -> density lower? performance optimization

        if behaviour["traffic_density"] > 0:
            behaviour["traffic_density"] = log_uniform_float(0.05, 0.90)

        if behaviour["traffic_density"] == 0.0:
            behaviour["need_inverse_traffic"] = False

        if behaviour["accident_prob"] > 0:
            behaviour["accident_prob"] = round(
                max(0.05, min(1.0, behaviour["accident_prob"] + random.uniform(-0.12, 0.12))), 2
            )

        road["lane_width"] = round(max(1.0, min(5.0, road["lane_width"] + random.uniform(-0.25, 0.25))), 2)
        road["exit_length"] = max(25, min(100, road["exit_length"] + random.choice([-10, -5, 0, 5, 10, 15])))

        if road["type"] == "block_sequence" and random.random() < 0.2:
            length = log_uniform_int(2, 25)
            road["config"] = "".join(
                random.choices(["S", "C", "O", "T", "r", "R"], weights=[5, 3, 2, 2, 1, 1], k=length)
            )

        target_json = {
            "start_seed": start_seed, # null or random [1,10.000]
            "random_lane_width": road["random_lane_width"],
            "random_lane_num": road["random_lane_num"],
            "traffic_density": behaviour["traffic_density"],
            "need_inverse_traffic": behaviour["need_inverse_traffic"],
            "traffic_mode": behaviour["traffic_mode"],
            "accident_prob": behaviour["accident_prob"],
            "static_traffic_object": behaviour["static_traffic_object"],
            #"pedestrian": random.choice([True, False]),
            #"cyclist": random.choice([True, False]),
            # traffic lights? 
            "map_config": {
                "type": road["type"],
                "config": road["config"],
                "lane_num": road["lane_num"],
                "lane_width": road["lane_width"],
                "exit_length": road["exit_length"],
            },
            "vehicle_config": {
                "no_wheel_friction": random.random() < 0.15, 
            },
        }

        data_sets.append(target_json)
    return data_sets
    
def main():
    print("Enter amount of json datasets that should be created")
    amount_json = int (sys.stdin.readline().strip())
    data_sets = generate_random_datasets(amount_json)

    # define output file 
    project_root = Path(__file__).resolve().parents[3]
    output_file = project_root / "data" / "training_data" / "llm_config_trainset.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # open or create output_file, "w" = overwrite 
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data_sets:
            f.write(json.dumps(item) + "\n") # convert python dictionary into correct json string

    print("Successfully created Jsonln output")

if __name__ == "__main__":
    main()