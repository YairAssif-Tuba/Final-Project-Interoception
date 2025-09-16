import os, json

def patch_time_bisection_hp(base_dir="enhanced_piezo_time_bisection_results"):
    """
    Add 'rnn_type': 'RNN' to hp.json files for time_bisection runs if missing.
    """
    for root, dirs, files in os.walk(base_dir):
        if "hp.json" in files:
            path = os.path.join(root, "hp.json")
            try:
                with open(path, "r") as f:
                    hp = json.load(f)
                if "rule_name" in hp and "bisection" in hp["rule_name"].lower():
                    if "rnn_type" not in hp:
                        hp["rnn_type"] = "RNN"
                        with open(path, "w") as f:
                            json.dump(hp, f, indent=2)
                        print(f"Patched {path}")
            except Exception as e:
                print(f"Skipping {path}: {e}")

# Run once
patch_time_bisection_hp("enhanced_piezo_time_bisection_results")
