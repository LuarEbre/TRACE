import json

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
"""
def generate_random_datasets():
    print()
    
def main():
    # prompt for llm model 
    SYSTEM_PROMPT = (
        "You are a MetaDrive Generator. Translate the user description into a valid JSON configuration."
        "Output ONLY raw JSON without markdown formatting or explanations. "
        "You will either use "
        "1) 'type': 'block_sequence' with 'config' as a string of block codes (S=Straight, C=Curve, O=Roundabout, T=T-Intersection, r=In-Ramp, R=Out-Ramp) when specific road elements are requested, OR "
        "2) 'type': 'block_num' with 'config' as an integer (e.g. 5) when a random or unspecified road layout is requested."
        "Default start_seed random between 1 and 1000 and num_scenarios to 1 unless specified otherwise."
    )

    # python list with dictionary elements as templates
    # (random)lane_width, (random)lane_num, exit_length, type (block_seq or num), config (string/int)
    ROAD_TEMPLATES = []

    # need_inverse_traffic, traffic_density, traffic_mode, accident_prob, random_traffic, static_traffic_object
    DRIVING_BEHAVIOUR = []

    # start_seed, num_scenarios
    INITIALIZATION = []


if __name__ == "__main__":
    main()