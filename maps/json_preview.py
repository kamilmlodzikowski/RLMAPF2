import json
import os
import numpy as np
import argparse

def preview_json(map_path, variants=None):
    data = json.load(open(map_path, "r"))
    map_width = data["metadata"]["width"]
    map_height = data["metadata"]["height"]

    if variants is not None:
        variants = [str(variant) for variant in variants]
    else:
        variants = data["map_variant"].keys()
    for map_variant_nr in variants:
        map_variant = data["map_variant"][map_variant_nr]
        map_array = np.array([[' ' for _ in range(map_width)] for _ in range(map_height)])
        for obstacle in map_variant["obstacles"]:
            map_array[obstacle[1]][obstacle[0]] = 'X'
    
        for j, starting_position in map_variant["starting_positions"].items():
            map_array[starting_position[1]][starting_position[0]] = chr(ord("A")+int(j))
        
        for j, goal_position in map_variant["goal_positions"].items():
            map_array[goal_position[1]][goal_position[0]] = str(j)

        # Replace array with colors
        map_array = np.where(map_array == 'X', "\033[90mX\033[00m", map_array)
        for i in range(10):
            color_i = 91 + i
            color1 = f"\033[{color_i}m"
            color2 = "\033[00m"
            map_array = np.where(map_array == str(i), color1+str(i)+color2, map_array)
            map_array = np.where(map_array == chr(ord("A")+i), color1+chr(ord("A")+i)+color2, map_array)
            map_array = np.where(map_array == chr(ord("a")+i), color1+chr(ord("a")+i)+color2, map_array)

        map_string = "\n".join(["".join(row) for row in map_array])
        
        print(str(map_variant_nr)+":")
        print(map_string+"\n")

    print("Maps size:")
    print(f"Width: {map_width}")
    print(f"Height: {map_height}")
    print(f"\nNumber of variants: {len(data['map_variant'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preview json map')
    parser.add_argument('map_path', type=str, help='Path to the map json file')
    parser.add_argument('--variant', type=int, nargs='+', help='Variant number to preview')
    
    args = parser.parse_args()
    map_path = args.map_path
    variants = args.variant

    # Check if file exists
    if not os.path.exists(map_path):
        # Try with cwd
        print(f"Warning: File {map_path} does not exist. Trying with {os.path.join(os.getcwd(), map_path)} instead.")
        if os.path.exists(os.path.join(os.getcwd(), map_path)):
            new_map_path = os.path.join(os.getcwd(), map_path)
            print(f"Warning: File {map_path} does not exist. Using {new_map_path} instead.")
            map_path = new_map_path

        else:
            print(f"Error: File {map_path} does not exist. Check if this is the correct file and doesn't start with a '/'.")
            exit()

    preview_json(map_path, variants)