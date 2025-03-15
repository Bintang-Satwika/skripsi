    file_path= os.path.join(CURRENT_DIR, "F_Testing_energy_F.json")
    with open(file_path, "w") as f:
        json.dump(energy, f, indent=4)
