import json
import os
import numpy as np
from json_preview import preview_json
import argparse

def generate_json(map_name, agents_min, agents_max, map_string, augment=[]):
    # Load template json file
    data = json.load(open(os.getcwd()+"/maps/_template.json"))

    # Remove leading newline character if empty
    if len(map_string.split("\n")[0]) == 0:
        map_string = map_string[1:]
    # Remove last newline character if empty
    if len(map_string.split("\n")[-1]) == 0:
        map_string = map_string[:-1]

    # Update map metadata
    data["metadata"]["name"] = map_name
    data["metadata"]["min_num_of_agents"] = agents_min
    data["metadata"]["max_num_of_agents"] = agents_max
    data["metadata"]["width"] = len(map_string.split("\n")[0])
    data["metadata"]["height"] = len(map_string.split("\n"))

    data["map_raw"] = map_string.split("\n")

    # Convert map string to uppercase to prevent replacing lowercase letters
    map_string = map_string.upper()

    map_strings = [map_string]

    if len(augment) > 0:

        # Add flipped horizontally version of the map
        if "flip_h" in augment:
            m_string = "\n".join([row for row in map_string.split("\n")][::-1])
            if m_string not in map_strings:
                map_strings.append(m_string)

        # Add flipped vertically version of the map
        if "flip_v" in augment:
            m_string = "\n".join([row[::-1] for row in map_string.split("\n")])
            if m_string not in map_strings:
                map_strings.append(m_string)

        # Add flipped horizontally and vertically version of the map
        if "flip_hv" in augment:
            m_string = "\n".join([row[::-1] for row in map_string.split("\n")][::-1])
            if m_string not in map_strings:
                map_strings.append(m_string)

        # Translate the map in x and y axis
        if "translate" in augment:
            old_map_strings = map_strings.copy()
            for i, map_string in enumerate(old_map_strings):
                # While all symbols at top are X, move top row to bottom
                while all([cell == "X" for cell in map_string.split("\n")[0]]):
                    map_string = "\n".join(map_string.split("\n")[1:]+[map_string.split("\n")[0]])
                    if map_string not in map_strings:
                        map_strings.append(map_string)

            old_map_strings = map_strings.copy()
            for i, map_string in enumerate(old_map_strings):
                # While all symbols at bottom are X, move bottom row to top
                while all([cell == "X" for cell in map_string.split("\n")[-1]]):
                    map_string = "\n".join([map_string.split("\n")[-1]]+map_string.split("\n")[:-1])
                    if map_string not in map_strings:
                        map_strings.append(map_string)

            old_map_strings = map_strings.copy()
            for i, map_string in enumerate(old_map_strings):
                # While all symbols at left are X, move left column to right
                while all([row[0] == "X" for row in map_string.split("\n")]):
                    map_string = "\n".join([row[1:]+row[0] for row in map_string.split("\n")])
                    if map_string not in map_strings:
                        map_strings.append(map_string)

            old_map_strings = map_strings.copy()
            for i, map_string in enumerate(old_map_strings):
                # While all symbols at right are X, move right column to left
                while all([row[-1] == "X" for row in map_string.split("\n")]):
                    map_string = "\n".join([row[-1]+row[:-1] for row in map_string.split("\n")])
                    if map_string not in map_strings:
                        map_strings.append(map_string)
        
        old_map_strings = map_strings.copy()

        # Swap number and letter positions
        if "swap" in augment:
            for map_string in old_map_strings:
                for i in range(10):
                    map_string = map_string.replace(str(i), chr(ord("a")+i))
                    map_string = map_string.replace(chr(ord("A")+i), str(i))
                    map_string = map_string.upper()
                if map_string not in map_strings:
                    map_strings.append(map_string)

        # Rotate the map 90 degrees
        old_map_strings = map_strings.copy()
        if "rotate90" in augment:
            for map_string in old_map_strings:
                map_string = "\n".join(["".join(row) for row in np.rot90(np.array([list(row) for row in map_string.split("\n")]))])
                if map_string not in map_strings:
                    map_strings.append(map_string)
        
        # Rotate the map 180 degrees
        old_map_strings = map_strings.copy()
        if "rotate180" in augment:
            for map_string in old_map_strings:
                map_string = "\n".join(["".join(row) for row in np.rot90(np.array([list(row) for row in map_string.split("\n")]), 2)])
                if map_string not in map_strings:
                    map_strings.append(map_string)

        # Rotate the map 270 degrees
        old_map_strings = map_strings.copy()
        if "rotate270" in augment:
            for map_string in old_map_strings:
                map_string = "\n".join(["".join(row) for row in np.rot90(np.array([list(row) for row in map_string.split("\n")]), 3)])
                if map_string not in map_strings:
                    map_strings.append(map_string)

        # Remove duplicates
        map_strings = list(set(map_strings))

        del old_map_strings
    
    # for i, map_string in enumerate(map_strings):
    #     print(i,":\n"+map_string+"\n")

    for i, map_string in enumerate(map_strings):
        data["map_variant"][str(i)] = {}
        # Load obstacles
        obstacles = []
        for y, row in enumerate(map_string.split("\n")):
            for x, cell in enumerate(row):
                if cell == "X" or cell == "x":
                    obstacles.append([x, y])
        data["map_variant"][str(i)]["obstacles"] = obstacles

        # Load start positions
        start_positions = {}
        for y, row in enumerate(map_string.split("\n")):
            for x, cell in enumerate(row):
                if cell >= "A" and cell <= "K":
                    index = ord(cell) - ord("A")
                    start_positions[index] = [x, y]
                if cell >= "a" and cell <= "k":
                    index = ord(cell) - ord("a")
                    start_positions[index] = [x, y]

        print("Start positions:", start_positions)
        print('-'*20)
        data["map_variant"][str(i)]["starting_positions"] = start_positions

        # Load goal positions
        goal_positions = {}
        for y, row in enumerate(map_string.split("\n")):
            for x, cell in enumerate(row):
                if cell >= "0" and cell <= "9":
                    goal_positions[int(cell)] = [x, y]
        data["map_variant"][str(i)]["goal_positions"] = goal_positions 

    start_positions_len = len(data["map_variant"]["0"]["starting_positions"])
    goal_positions_len = len(data["map_variant"]["0"]["goal_positions"])

    # Check if number of agents and goals match
    if start_positions_len != goal_positions_len:
        raise ValueError(f"Number of agents ({start_positions_len}) does not match number of goals ({goal_positions_len})")
    
    # Check if number of agents matches the range
    if start_positions_len != 0:
        if start_positions_len != agents_min or start_positions_len != agents_max:
            # print("Map:")
            # print(map_string)
            # print("Start positions:")
            # print(data["map_variant"]["0"]["starting_positions"])
            raise ValueError(f"Number of agents ({start_positions_len}) does not match the range ({agents_min}-{agents_max})")
        
    # Define filename
    filename = map_name + "_" + str(agents_min) + "-" + str(agents_max) + "a-" + str(data["metadata"]["width"]) + "x" + str(data["metadata"]["height"]) + ".json"

    if(not os.path.exists(os.getcwd()+"/maps")):
        os.makedirs(os.getcwd()+"/maps")
    if(os.path.exists(os.getcwd()+f"/maps/{filename}")):
        # Ask for confirmation to overwrite
        print(f"File {filename}.json already exists. Do you want to overwrite it? (y/n)")
        if input().lower() != "y":
            return False
        # Create backup
        os.rename(os.getcwd()+f"/maps/{filename}", os.getcwd()+f"/maps/backup.json")
    json.dump(data, open(os.getcwd()+f"/maps/{filename}", "w"), indent=4)
    if(os.path.exists(os.getcwd()+f"/maps/{filename}")):
        return os.getcwd()+"/maps/"+filename
    else:
        return False

if __name__ == "__main__":
    # Parse arguments
    args_parser = argparse.ArgumentParser(description="Generate json file from map string", 
                                          usage="python json_generator.py -n <name> -a <agents>\
                                            [--flip_h] [--flip_v] [--flip_hv] [--translate] [--swap] [--rotate90] [--rotate180] [--rotate270]")
                                          
    
    args_parser.add_argument("-n", "--name", help="Name of the map (json file will be named differently)", type=str, required=True)
    args_parser.add_argument("-a", "--agents", help="Number of agents (min-max)", type=str, required=True)
    args_parser.add_argument("--flip_h", help="Flip map horizontally", action="store_true")
    args_parser.add_argument("--flip_v", help="Flip map vertically", action="store_true")
    args_parser.add_argument("--flip_hv", help="Flip map horizontally and vertically", action="store_true")
    args_parser.add_argument("--translate", help="Translate map in x and y axis", action="store_true")
    args_parser.add_argument("--swap", help="Swap number and letter positions", action="store_true")
    args_parser.add_argument("--rot90", help="Rotate map 90 degrees", action="store_true")
    args_parser.add_argument("--rot180", help="Rotate map 180 degrees", action="store_true")
    args_parser.add_argument("--rot270", help="Rotate map 270 degrees", action="store_true")
    args_parser.add_argument("-m", "--map_txt_file", help="Path to the map txt file, default is maps/json_generator_map.txt", type=str, default="maps/json_generator_map.txt")



    args = args_parser.parse_args()
    
    map_name = args.name
    agents = args.agents
    augment = []
    if args.flip_h:
        augment.append("flip_h")
    if args.flip_v:
        augment.append("flip_v")
    if args.flip_hv:
        augment.append("flip_hv")
    if args.translate:
        augment.append("translate")
    if args.swap:
        augment.append("swap")
    if args.rot90:
        augment.append("rotate90")
    if args.rot180:
        augment.append("rotate180")
    if args.rot270:
        augment.append("rotate270")

    # Parse agents range
    if "-" not in agents:
        print("Invalid agents range! Use format min-max (e.g. 1-4)")
        exit()
    agents_min, agents_max = agents.split("-")
    agents_min = int(agents_min)
    agents_max = int(agents_max)
    if agents_min > agents_max:
        agents_min, agents_max = agents_max, agents_min
    
    # Generate json file
    with open(os.getcwd()+"/maps/json_generator_map.txt", "r") as f:
        map_string = f.read()
        map_path = generate_json(map_name, agents_min, agents_max, map_string, augment)
        if map_path:
            preview_json(map_path)
            print("Json file generated successfully!")
        else:
            print("Failed to generate json file!")

# TEMPLATE, ABC for agents, 012 for goals, X for obstacles, space for empty
"""
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
XXXXXXXXXXXXXXXXXXXXX
"""