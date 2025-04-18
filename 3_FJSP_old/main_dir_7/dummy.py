data = {
    "rewards": {
        "1": 108.14814814814794,
        "2": 86.475,
        "3": 142.17870370370338,
        "4": 125.98425925925926,
        "5": 131.29537037036994,
        "6": 142.306481481481,
        "7": 132.22777777777753,
        "8": 148.86574074074045,
        "9": 90.0305555555557,
        "10": 132.9324074074072,
        "11": 145.79999999999959,
        "12": 121.95648148148113,
        "13": 126.88055555555553,
        "14": 156.40370370370357,
        "15": 98.82777777777784,
        "16": 129.27129629629633,
        "17": 131.31203703703682,
        "18": 104.81296296296294,
        "19": 110.29814814814814,
        "20": 63.049999999999976
    },
    "makespan": {
        "1": 418,
        "2": 439,
        "3": 389,
        "4": 403,
        "5": 397,
        "6": 388,
        "7": 398,
        "8": 384,
        "9": 436,
        "10": 396,
        "11": 384,
        "12": 408,
        "13": 401,
        "14": 377,
        "15": 429,
        "16": 401,
        "17": 398,
        "18": 425,
        "19": 417,
        "20": 462
    },
    "energy": {
        "1": 656,
        "2": 659,
        "3": 622,
        "4": 637,
        "5": 639,
        "6": 628,
        "7": 629,
        "8": 618,
        "9": 647,
        "10": 635,
        "11": 633,
        "12": 630,
        "13": 641,
        "14": 610,
        "15": 638,
        "16": 629,
        "17": 632,
        "18": 627,
        "19": 636,
        "20": 647
    }
}

# Membagi energy / makespan untuk setiap key
energy_per_makespan = {
    key: data["energy"][key] / data["makespan"][key]
    for key in data["energy"]
}

# Output hasilnya
for key, value in energy_per_makespan.items():
    print(f"{key}: {value:.4f}")
