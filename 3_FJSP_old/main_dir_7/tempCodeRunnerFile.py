    combined_data = {
        "rewards": rewards,
        "makespan": makespan,
        "energy": energy
    }

    # Write the combined dictionary to a single JSON file
    file_path = os.path.join(CURRENT_DIR, "Testing_Random.json")
    with open(file_path, "w") as f:
        json.dump(combined_data, f, indent=4)
