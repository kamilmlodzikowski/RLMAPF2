import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from json_preview import preview_json

Coordinate = Tuple[int, int]

EMPTY_CELL = " "
OBSTACLE_CELL = "X"

DEFAULT_WIDTH = 20
DEFAULT_HEIGHT = 20
CELL_SIZE = 28

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "_template.json")
DEFAULT_OUTPUT_DIR = BASE_DIR


@dataclass
class VariantData:
    grid: List[List[str]]
    starts: Dict[int, Coordinate]
    goals: Dict[int, Coordinate]

    def clone(self) -> "VariantData":
        return VariantData(
            [row[:] for row in self.grid],
            {agent: (pos[0], pos[1]) for agent, pos in self.starts.items()},
            {agent: (pos[0], pos[1]) for agent, pos in self.goals.items()},
        )


def variant_key(variant: VariantData) -> Tuple:
    grid_key = tuple("".join(row) for row in variant.grid)
    starts_key = tuple(sorted((int(agent), pos[0], pos[1]) for agent, pos in variant.starts.items()))
    goals_key = tuple(sorted((int(agent), pos[0], pos[1]) for agent, pos in variant.goals.items()))
    return grid_key, starts_key, goals_key


def flip_horizontal_variant(variant: VariantData) -> VariantData:
    height = len(variant.grid)
    new_grid = [row[:] for row in reversed(variant.grid)]
    new_starts = {agent: (pos[0], height - 1 - pos[1]) for agent, pos in variant.starts.items()}
    new_goals = {agent: (pos[0], height - 1 - pos[1]) for agent, pos in variant.goals.items()}
    return VariantData(new_grid, new_starts, new_goals)


def flip_vertical_variant(variant: VariantData) -> VariantData:
    width = len(variant.grid[0]) if variant.grid else 0
    new_grid = [list(reversed(row)) for row in variant.grid]
    new_starts = {agent: (width - 1 - pos[0], pos[1]) for agent, pos in variant.starts.items()}
    new_goals = {agent: (width - 1 - pos[0], pos[1]) for agent, pos in variant.goals.items()}
    return VariantData(new_grid, new_starts, new_goals)


def rotate90_variant(variant: VariantData) -> VariantData:
    old_height = len(variant.grid)
    old_width = len(variant.grid[0]) if old_height else 0
    if old_height == 0 or old_width == 0:
        return variant.clone()
    new_grid = [
        [variant.grid[y][x] for y in range(old_height)]
        for x in range(old_width - 1, -1, -1)
    ]
    new_starts = {
        agent: (pos[1], old_width - 1 - pos[0]) for agent, pos in variant.starts.items()
    }
    new_goals = {
        agent: (pos[1], old_width - 1 - pos[0]) for agent, pos in variant.goals.items()
    }
    return VariantData(new_grid, new_starts, new_goals)


def rotate270_variant(variant: VariantData) -> VariantData:
    old_height = len(variant.grid)
    old_width = len(variant.grid[0]) if old_height else 0
    if old_height == 0 or old_width == 0:
        return variant.clone()
    new_grid = [
        [variant.grid[y][x] for y in range(old_height - 1, -1, -1)]
        for x in range(old_width)
    ]
    new_starts = {
        agent: (old_height - 1 - pos[1], pos[0])
        for agent, pos in variant.starts.items()
    }
    new_goals = {
        agent: (old_height - 1 - pos[1], pos[0])
        for agent, pos in variant.goals.items()
    }
    return VariantData(new_grid, new_starts, new_goals)


def rotate180_variant(variant: VariantData) -> VariantData:
    width = len(variant.grid[0]) if variant.grid else 0
    height = len(variant.grid)
    new_grid = [list(reversed(row)) for row in reversed(variant.grid)]
    new_starts = {
        agent: (width - 1 - pos[0], height - 1 - pos[1])
        for agent, pos in variant.starts.items()
    }
    new_goals = {
        agent: (width - 1 - pos[0], height - 1 - pos[1])
        for agent, pos in variant.goals.items()
    }
    return VariantData(new_grid, new_starts, new_goals)


def swap_positions_variant(variant: VariantData) -> VariantData:
    if set(variant.starts.keys()) != set(variant.goals.keys()):
        raise ValueError("Cannot swap positions when start and goal agent sets differ.")
    new_grid = [row[:] for row in variant.grid]
    new_starts = {agent: variant.goals[agent] for agent in variant.starts.keys()}
    new_goals = {agent: variant.starts[agent] for agent in variant.goals.keys()}
    return VariantData(new_grid, new_starts, new_goals)


def row_all_obstacles(row: List[str]) -> bool:
    return all(cell == OBSTACLE_CELL for cell in row)


def col_all_obstacles(grid: List[List[str]], idx: int) -> bool:
    return all(row[idx] == OBSTACLE_CELL for row in grid)


def shift_up_variant(variant: VariantData) -> VariantData:
    height = len(variant.grid)
    if height == 0:
        return variant.clone()
    new_grid = [row[:] for row in variant.grid[1:]] + [variant.grid[0][:]]
    new_starts = {agent: (pos[0], (pos[1] - 1) % height) for agent, pos in variant.starts.items()}
    new_goals = {agent: (pos[0], (pos[1] - 1) % height) for agent, pos in variant.goals.items()}
    return VariantData(new_grid, new_starts, new_goals)


def shift_down_variant(variant: VariantData) -> VariantData:
    height = len(variant.grid)
    if height == 0:
        return variant.clone()
    new_grid = [variant.grid[-1][:]] + [row[:] for row in variant.grid[:-1]]
    new_starts = {agent: (pos[0], (pos[1] + 1) % height) for agent, pos in variant.starts.items()}
    new_goals = {agent: (pos[0], (pos[1] + 1) % height) for agent, pos in variant.goals.items()}
    return VariantData(new_grid, new_starts, new_goals)


def shift_left_variant(variant: VariantData) -> VariantData:
    width = len(variant.grid[0]) if variant.grid else 0
    if width == 0:
        return variant.clone()
    new_grid = [row[1:] + row[:1] for row in variant.grid]
    new_starts = {agent: ((pos[0] - 1) % width, pos[1]) for agent, pos in variant.starts.items()}
    new_goals = {agent: ((pos[0] - 1) % width, pos[1]) for agent, pos in variant.goals.items()}
    return VariantData(new_grid, new_starts, new_goals)


def shift_right_variant(variant: VariantData) -> VariantData:
    width = len(variant.grid[0]) if variant.grid else 0
    if width == 0:
        return variant.clone()
    new_grid = [row[-1:] + row[:-1] for row in variant.grid]
    new_starts = {agent: ((pos[0] + 1) % width, pos[1]) for agent, pos in variant.starts.items()}
    new_goals = {agent: ((pos[0] + 1) % width, pos[1]) for agent, pos in variant.goals.items()}
    return VariantData(new_grid, new_starts, new_goals)


def translate_variants(base_variant: VariantData) -> List[VariantData]:
    """Replicates the legacy translate augmentation by cycling obstacle-only borders."""
    results: List[VariantData] = []
    visited = {variant_key(base_variant)}
    queue = deque([base_variant])

    while queue:
        current = queue.popleft()
        if not current.grid:
            continue
        width = len(current.grid[0])
        height = len(current.grid)
        operations = []
        if row_all_obstacles(current.grid[0]):
            operations.append(shift_up_variant)
        if row_all_obstacles(current.grid[-1]):
            operations.append(shift_down_variant)
        if width > 0 and col_all_obstacles(current.grid, 0):
            operations.append(shift_left_variant)
        if width > 0 and col_all_obstacles(current.grid, width - 1):
            operations.append(shift_right_variant)

        for op in operations:
            candidate = op(current)
            key = variant_key(candidate)
            if key not in visited:
                visited.add(key)
                results.append(candidate)
                queue.append(candidate)

    return results


def generate_json(
    map_name: str,
    agents_min: int,
    agents_max: int,
    grid: List[List[str]],
    start_positions: Dict[int, Coordinate],
    goal_positions: Dict[int, Coordinate],
    augmentations: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    if augmentations is None:
        augmentations = []

    if not grid or not grid[0]:
        raise ValueError("Grid cannot be empty.")

    width = len(grid[0])
    height = len(grid)

    for row in grid:
        if len(row) != width:
            raise ValueError("All grid rows must have the same width.")
        if any(cell not in (EMPTY_CELL, OBSTACLE_CELL) for cell in row):
            raise ValueError("Grid cells must be spaces or 'X'.")

    starts = {int(agent): (int(pos[0]), int(pos[1])) for agent, pos in start_positions.items()}
    goals = {int(agent): (int(pos[0]), int(pos[1])) for agent, pos in goal_positions.items()}

    if set(starts.keys()) != set(goals.keys()):
        raise ValueError("Start and goal positions must be defined for the same agents.")

    for agent, (x, y) in starts.items():
        if not (0 <= x < width) or not (0 <= y < height):
            raise ValueError(f"Start position for agent {agent} is outside the grid.")
    for agent, (x, y) in goals.items():
        if not (0 <= x < width) or not (0 <= y < height):
            raise ValueError(f"Goal position for agent {agent} is outside the grid.")

    base_variant = VariantData(
        [row[:] for row in grid],
        {agent: (pos[0], pos[1]) for agent, pos in starts.items()},
        {agent: (pos[0], pos[1]) for agent, pos in goals.items()},
    )

    variants: List[VariantData] = [base_variant]
    seen = {variant_key(base_variant)}

    for augment in augmentations:
        existing = list(variants)
        for variant in existing:
            if augment == "flip_h":
                candidates = [flip_horizontal_variant(variant)]
            elif augment == "flip_v":
                candidates = [flip_vertical_variant(variant)]
            elif augment == "flip_hv":
                candidates = [flip_vertical_variant(flip_horizontal_variant(variant))]
            elif augment == "translate":
                candidates = translate_variants(variant)
            elif augment == "swap":
                candidates = [swap_positions_variant(variant)]
            elif augment == "rotate90":
                candidates = [rotate90_variant(variant)]
            elif augment == "rotate180":
                candidates = [rotate180_variant(variant)]
            elif augment == "rotate270":
                candidates = [rotate270_variant(variant)]
            else:
                continue

            for candidate in candidates:
                key = variant_key(candidate)
                if key not in seen:
                    seen.add(key)
                    variants.append(candidate)

    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"Template file not found: {TEMPLATE_PATH}")
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as template_file:
        data = json.load(template_file)

    data["metadata"]["name"] = map_name
    data["metadata"]["min_num_of_agents"] = agents_min
    data["metadata"]["max_num_of_agents"] = agents_max
    data["metadata"]["width"] = width
    data["metadata"]["height"] = height
    data["map_raw"] = ["".join(row) for row in grid]
    data["map_variant"] = {}

    for idx, variant in enumerate(variants):
        obstacles = [
            [x, y]
            for y, row in enumerate(variant.grid)
            for x, cell in enumerate(row)
            if cell == OBSTACLE_CELL
        ]
        start_block = {
            str(agent): [pos[0], pos[1]] for agent, pos in sorted(variant.starts.items())
        }
        goal_block = {
            str(agent): [pos[0], pos[1]] for agent, pos in sorted(variant.goals.items())
        }
        data["map_variant"][str(idx)] = {
            "obstacles": obstacles,
            "starting_positions": start_block,
            "goal_positions": goal_block,
        }

    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{map_name}_{agents_min}-{agents_max}a-{width}x{height}.json"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        if not overwrite:
            raise FileExistsError(output_path)
        backup_path = output_path + ".bak"
        try:
            os.replace(output_path, backup_path)
        except OSError:
            pass

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=4)

    return output_path


class MapEditorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("RLMAPF2 JSON Map Generator")

        self.cell_size = CELL_SIZE
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.grid = [[EMPTY_CELL for _ in range(self.width)] for _ in range(self.height)]

        self.agent_data: Dict[int, Dict[str, Optional[Coordinate]]] = {}
        self.agent_list: List[int] = []
        self.next_agent_id = 0

        self.mode_var = tk.StringVar(value="obstacle")
        self.map_name_var = tk.StringVar(value="new_map")
        self.agent_min_var = tk.StringVar(value="0")
        self.agent_max_var = tk.StringVar(value="0")
        self.width_var = tk.StringVar(value=str(self.width))
        self.height_var = tk.StringVar(value=str(self.height))
        self.output_dir_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.preview_var = tk.BooleanVar(value=False)
        self.agent_info_var = tk.StringVar(value="No agents defined. Use Add to create one.")

        self.augment_vars = {
            "flip_h": tk.BooleanVar(value=False),
            "flip_v": tk.BooleanVar(value=False),
            "flip_hv": tk.BooleanVar(value=False),
            "translate": tk.BooleanVar(value=False),
            "swap": tk.BooleanVar(value=False),
            "rotate90": tk.BooleanVar(value=False),
            "rotate180": tk.BooleanVar(value=False),
            "rotate270": tk.BooleanVar(value=False),
        }

        self.canvas: Optional[tk.Canvas] = None
        self.rectangles: List[List[int]] = []
        self.cell_text: List[List[int]] = []
        self.control_canvas: Optional[tk.Canvas] = None
        self.canvas_scroll_x: Optional[ttk.Scrollbar] = None
        self.canvas_scroll_y: Optional[ttk.Scrollbar] = None
        self.canvas_max_width = 800
        self.canvas_max_height = 600

        self._build_ui()
        self.rebuild_canvas()
        self.refresh_canvas()

    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        control_container = ttk.Frame(self.root)
        control_container.grid(row=0, column=0, sticky="ns")
        control_container.rowconfigure(0, weight=1)
        control_container.columnconfigure(0, weight=1)

        self.control_canvas = tk.Canvas(control_container, highlightthickness=0)
        self.control_canvas.grid(row=0, column=0, sticky="nsew")
        control_scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=self.control_canvas.yview)
        control_scrollbar.grid(row=0, column=1, sticky="ns")
        self.control_canvas.configure(yscrollcommand=control_scrollbar.set)
        self.control_canvas.configure(width=320)

        control_frame = ttk.Frame(self.control_canvas, padding=10)
        control_window = self.control_canvas.create_window((0, 0), window=control_frame, anchor="nw")

        def _control_frame_configure(_event: tk.Event) -> None:
            if self.control_canvas is None:
                return
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))
            self.control_canvas.itemconfigure(control_window, width=self.control_canvas.winfo_width())

        control_frame.bind("<Configure>", _control_frame_configure)
        self.control_canvas.bind(
            "<Configure>",
            lambda event: self.control_canvas.itemconfigure(control_window, width=event.width),
        )

        canvas_outer = ttk.Frame(self.root, padding=10)
        canvas_outer.grid(row=0, column=1, sticky="nsew")
        canvas_outer.rowconfigure(0, weight=1)
        canvas_outer.columnconfigure(0, weight=1)

        canvas_container = ttk.Frame(canvas_outer)
        canvas_container.grid(row=0, column=0, sticky="nsew")
        canvas_container.rowconfigure(0, weight=1)
        canvas_container.columnconfigure(0, weight=1)

        self.canvas_scroll_y = ttk.Scrollbar(canvas_container, orient="vertical")
        self.canvas_scroll_y.grid(row=0, column=1, sticky="ns")
        self.canvas_scroll_x = ttk.Scrollbar(canvas_container, orient="horizontal")
        self.canvas_scroll_x.grid(row=1, column=0, sticky="ew")

        self.canvas = tk.Canvas(
            canvas_container,
            background="white",
            highlightthickness=1,
            highlightbackground="#bdbdbd",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.configure(xscrollcommand=self.canvas_scroll_x.set, yscrollcommand=self.canvas_scroll_y.set)
        self.canvas_scroll_x.configure(command=self.canvas.xview)
        self.canvas_scroll_y.configure(command=self.canvas.yview)

        self.canvas.bind("<Button-1>", self.on_canvas_left_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.canvas.bind("<Enter>", self._on_canvas_enter)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        if self.control_canvas is not None:
            self.control_canvas.bind("<Enter>", self._on_control_enter)
            self.control_canvas.bind("<Leave>", self._on_control_leave)
            control_frame.bind("<Enter>", self._on_control_enter)
            control_frame.bind("<Leave>", self._on_control_leave)
        control_frame.columnconfigure(0, weight=1)

        meta_frame = ttk.LabelFrame(control_frame, text="Map Metadata", padding=8)
        meta_frame.grid(row=0, column=0, sticky="ew")
        meta_frame.columnconfigure(1, weight=1)

        ttk.Label(meta_frame, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(meta_frame, textvariable=self.map_name_var).grid(row=0, column=1, sticky="ew")

        ttk.Label(meta_frame, text="Agents (min/max)").grid(row=1, column=0, sticky="w", pady=(6, 0))
        agent_range_frame = ttk.Frame(meta_frame)
        agent_range_frame.grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Entry(agent_range_frame, width=5, textvariable=self.agent_min_var).pack(side="left", padx=(0, 4))
        ttk.Entry(agent_range_frame, width=5, textvariable=self.agent_max_var).pack(side="left")

        grid_frame = ttk.LabelFrame(control_frame, text="Grid Size", padding=8)
        grid_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(grid_frame, text="Width").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid_frame, width=6, textvariable=self.width_var).grid(row=0, column=1, sticky="w")
        ttk.Label(grid_frame, text="Height").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(grid_frame, width=6, textvariable=self.height_var).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Button(grid_frame, text="Resize", command=self.apply_size_change).grid(row=0, column=2, rowspan=2, padx=(12, 0))

        mode_frame = ttk.LabelFrame(control_frame, text="Edit Mode", padding=8)
        mode_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Radiobutton(mode_frame, text="Obstacle", value="obstacle", variable=self.mode_var).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode_frame, text="Empty", value="empty", variable=self.mode_var).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(mode_frame, text="Start (selected agent)", value="start", variable=self.mode_var).grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(mode_frame, text="Goal (selected agent)", value="goal", variable=self.mode_var).grid(row=3, column=0, sticky="w")
        ttk.Label(mode_frame, text="Left click: apply mode\nRight click: clear cell", foreground="#555").grid(row=4, column=0, sticky="w", pady=(6, 0))

        agents_frame = ttk.LabelFrame(control_frame, text="Agents", padding=8)
        agents_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        agents_frame.columnconfigure(0, weight=1)
        agents_frame.rowconfigure(0, weight=1)

        self.agent_listbox = tk.Listbox(agents_frame, height=6, exportselection=False)
        self.agent_listbox.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.agent_listbox.bind("<<ListboxSelect>>", self.on_select_agent)

        agent_btns = ttk.Frame(agents_frame)
        agent_btns.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        ttk.Button(agent_btns, text="Add", command=self.add_agent).grid(row=0, column=0, sticky="ew")
        ttk.Button(agent_btns, text="Remove", command=self.remove_agent).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(agent_btns, text="Set Start", command=lambda: self.set_mode("start")).grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(agent_btns, text="Set Goal", command=lambda: self.set_mode("goal")).grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(agent_btns, text="Clear Positions", command=self.clear_agent_positions).grid(row=4, column=0, sticky="ew", pady=(12, 0))

        ttk.Label(agents_frame, textvariable=self.agent_info_var, wraplength=180, justify="left").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )

        augment_frame = ttk.LabelFrame(control_frame, text="Augmentations", padding=8)
        augment_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        augment_names = [
            ("Flip Horizontal", "flip_h"),
            ("Flip Vertical", "flip_v"),
            ("Flip H & V", "flip_hv"),
            ("Translate", "translate"),
            ("Swap Start/Goal", "swap"),
            ("Rotate 90 deg", "rotate90"),
            ("Rotate 180 deg", "rotate180"),
            ("Rotate 270 deg", "rotate270"),
        ]
        for idx, (label, key) in enumerate(augment_names):
            ttk.Checkbutton(augment_frame, text=label, variable=self.augment_vars[key]).grid(row=idx // 2, column=idx % 2, sticky="w")

        output_frame = ttk.LabelFrame(control_frame, text="Output", padding=8)
        output_frame.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        output_frame.columnconfigure(0, weight=1)
        ttk.Entry(output_frame, textvariable=self.output_dir_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(output_frame, text="Browse...", command=self.browse_output_dir).grid(row=0, column=1, padx=(8, 0))
        ttk.Checkbutton(output_frame, text="Preview in console after save", variable=self.preview_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0)
        )

        ttk.Button(control_frame, text="Generate JSON", command=self.generate_map).grid(row=6, column=0, sticky="ew", pady=(16, 0))
        ttk.Button(control_frame, text="Clear Obstacles", command=self.clear_obstacles).grid(row=7, column=0, sticky="ew", pady=(6, 0))

    def set_mode(self, mode: str) -> None:
        self.mode_var.set(mode)
        if not self.agent_listbox.curselection() and self.agent_list:
            self.agent_listbox.selection_set(tk.END)
            self.agent_listbox.event_generate("<<ListboxSelect>>")

    def add_agent(self) -> None:
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        self.agent_data[agent_id] = {"start": None, "goal": None}
        self.agent_list.append(agent_id)
        self.agent_listbox.insert(tk.END, f"Agent {agent_id}")
        self.agent_listbox.selection_clear(0, tk.END)
        self.agent_listbox.selection_set(tk.END)
        self.agent_listbox.event_generate("<<ListboxSelect>>")
        self.update_agent_count_fields()

    def remove_agent(self) -> None:
        selection = self.agent_listbox.curselection()
        if not selection:
            messagebox.showinfo("Remove agent", "Select an agent to remove.")
            return
        idx = selection[0]
        agent_id = self.agent_list[idx]
        agent_positions = self.agent_data.pop(agent_id)
        del self.agent_list[idx]
        self.agent_listbox.delete(idx)
        for key in ("start", "goal"):
            pos = agent_positions[key]
            if pos:
                self.update_cell_display(pos[0], pos[1])
        if self.agent_list:
            new_idx = min(idx, len(self.agent_list) - 1)
            self.agent_listbox.selection_set(new_idx)
            self.agent_listbox.event_generate("<<ListboxSelect>>")
        else:
            self.agent_info_var.set("No agents defined. Use Add to create one.")
        self.update_agent_count_fields()

    def clear_agent_positions(self) -> None:
        agent_id = self.selected_agent()
        if agent_id is None:
            messagebox.showinfo("Clear positions", "Select an agent first.")
            return
        data = self.agent_data[agent_id]
        for key in ("start", "goal"):
            pos = data[key]
            if pos:
                data[key] = None
                self.update_cell_display(pos[0], pos[1])
        self.update_agent_info()

    def selected_agent(self) -> Optional[int]:
        selection = self.agent_listbox.curselection()
        if not selection:
            return None
        return self.agent_list[selection[0]]

    def on_select_agent(self, _event: Optional[tk.Event] = None) -> None:
        self.update_agent_info()

    def update_agent_info(self) -> None:
        agent_id = self.selected_agent()
        if agent_id is None:
            if not self.agent_list:
                self.agent_info_var.set("No agents defined. Use Add to create one.")
            else:
                self.agent_info_var.set("No agent selected.")
            return
        data = self.agent_data[agent_id]
        start_text = f"Start: {data['start']}" if data["start"] is not None else "Start: not set"
        goal_text = f"Goal: {data['goal']}" if data["goal"] is not None else "Goal: not set"
        self.agent_info_var.set(f"Agent {agent_id}\n{start_text}\n{goal_text}")

    def update_agent_count_fields(self) -> None:
        count = len(self.agent_data)
        self.agent_min_var.set(str(count))
        self.agent_max_var.set(str(count))

    def apply_size_change(self) -> None:
        try:
            new_width = int(self.width_var.get())
            new_height = int(self.height_var.get())
        except ValueError:
            messagebox.showerror("Invalid size", "Width and height must be integers.")
            return

        if new_width <= 0 or new_height <= 0:
            messagebox.showerror("Invalid size", "Width and height must be positive.")
            return

        self.resize_grid(new_width, new_height)

    def resize_grid(self, new_width: int, new_height: int) -> None:
        new_grid = [[EMPTY_CELL for _ in range(new_width)] for _ in range(new_height)]
        for y in range(min(self.height, new_height)):
            for x in range(min(self.width, new_width)):
                new_grid[y][x] = self.grid[y][x]
        self.grid = new_grid
        self.width = new_width
        self.height = new_height
        self.width_var.set(str(new_width))
        self.height_var.set(str(new_height))

        for agent in self.agent_data.values():
            for key in ("start", "goal"):
                pos = agent[key]
                if pos:
                    x, y = pos
                    if x >= new_width or y >= new_height:
                        agent[key] = None
        self.rebuild_canvas()
        self.refresh_canvas()
        self.update_agent_info()

    def rebuild_canvas(self) -> None:
        if self.canvas is None:
            return
        self.canvas.delete("all")
        self.rectangles = []
        self.cell_text = []
        for y in range(self.height):
            rect_row: List[int] = []
            text_row: List[int] = []
            for x in range(self.width):
                x0 = x * self.cell_size
                y0 = y * self.cell_size
                rect = self.canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + self.cell_size,
                    y0 + self.cell_size,
                    fill="white",
                    outline="#c0c0c0",
                )
                text = self.canvas.create_text(
                    x0 + self.cell_size / 2,
                    y0 + self.cell_size / 2,
                    text="",
                    fill="#2f3542",
                    font=("TkDefaultFont", 9),
                )
                rect_row.append(rect)
                text_row.append(text)
            self.rectangles.append(rect_row)
            self.cell_text.append(text_row)
        self.update_scroll_region()

    def refresh_canvas(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                self.update_cell_display(x, y)

    def update_scroll_region(self) -> None:
        if self.canvas is None:
            return
        content_width = self.width * self.cell_size
        content_height = self.height * self.cell_size
        view_width = max(1, min(content_width, self.canvas_max_width))
        view_height = max(1, min(content_height, self.canvas_max_height))
        self.canvas.configure(width=view_width, height=view_height)
        self.canvas.configure(scrollregion=(0, 0, content_width, content_height))

    @staticmethod
    def _scroll_units_from_delta(delta: int) -> int:
        if delta == 0:
            return 0
        magnitude = max(1, int(abs(delta) / 120))
        return -magnitude if delta > 0 else magnitude

    def _on_control_enter(self, _event: tk.Event) -> None:
        if self.control_canvas is None:
            return
        self.control_canvas.bind_all("<MouseWheel>", self._on_menu_mousewheel)
        self.control_canvas.bind_all("<Button-4>", self._on_menu_button_scroll)
        self.control_canvas.bind_all("<Button-5>", self._on_menu_button_scroll)

    def _on_control_leave(self, _event: tk.Event) -> None:
        if self.control_canvas is None:
            return
        self.control_canvas.unbind_all("<MouseWheel>")
        self.control_canvas.unbind_all("<Button-4>")
        self.control_canvas.unbind_all("<Button-5>")

    def _on_menu_mousewheel(self, event: tk.Event) -> None:
        if self.control_canvas is None:
            return
        step = self._scroll_units_from_delta(event.delta)
        if step != 0:
            self.control_canvas.yview_scroll(step, "units")

    def _on_menu_button_scroll(self, event: tk.Event) -> None:
        if self.control_canvas is None:
            return
        if event.num == 4:
            self.control_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.control_canvas.yview_scroll(1, "units")

    def _on_canvas_enter(self, _event: tk.Event) -> None:
        if self.canvas is None:
            return
        self.canvas.bind_all("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_canvas_button_scroll)
        self.canvas.bind_all("<Button-5>", self._on_canvas_button_scroll)

    def _on_canvas_leave(self, _event: tk.Event) -> None:
        if self.canvas is None:
            return
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_canvas_mousewheel(self, event: tk.Event) -> None:
        if self.canvas is None:
            return
        step = self._scroll_units_from_delta(event.delta)
        if step == 0:
            return
        if event.state & 0x0001:
            self.canvas.xview_scroll(step, "units")
        else:
            self.canvas.yview_scroll(step, "units")

    def _on_canvas_button_scroll(self, event: tk.Event) -> None:
        if self.canvas is None:
            return
        if event.num == 4:
            step = -1
        elif event.num == 5:
            step = 1
        else:
            return
        if event.state & 0x0001:
            self.canvas.xview_scroll(step, "units")
        else:
            self.canvas.yview_scroll(step, "units")

    def on_canvas_left_click(self, event: tk.Event) -> None:
        if self.canvas is None:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x = int(canvas_x // self.cell_size)
        y = int(canvas_y // self.cell_size)
        self.apply_mode_to_cell(x, y)

    def on_canvas_drag(self, event: tk.Event) -> None:
        if self.mode_var.get() not in ("obstacle", "empty"):
            return
        if self.canvas is None:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x = int(canvas_x // self.cell_size)
        y = int(canvas_y // self.cell_size)
        self.apply_mode_to_cell(x, y)

    def on_canvas_right_click(self, event: tk.Event) -> None:
        if self.canvas is None:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x = int(canvas_x // self.cell_size)
        y = int(canvas_y // self.cell_size)
        previous_mode = self.mode_var.get()
        self.mode_var.set("empty")
        self.apply_mode_to_cell(x, y)
        self.mode_var.set(previous_mode)

    def apply_mode_to_cell(self, x: int, y: int) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        mode = self.mode_var.get()

        if mode == "obstacle":
            self.grid[y][x] = OBSTACLE_CELL
            self.clear_cell_agents(x, y)
        elif mode == "empty":
            self.grid[y][x] = EMPTY_CELL
            self.clear_cell_agents(x, y)
        elif mode in ("start", "goal"):
            agent_id = self.selected_agent()
            if agent_id is None:
                messagebox.showinfo("Assign position", "Select an agent before assigning start/goal.")
                return
            if self.grid[y][x] == OBSTACLE_CELL:
                self.grid[y][x] = EMPTY_CELL
            self.assign_agent_position(agent_id, mode, x, y)
        self.update_cell_display(x, y)

    def clear_cell_agents(self, x: int, y: int) -> None:
        updated = False
        for agent_id, data in self.agent_data.items():
            for key in ("start", "goal"):
                if data[key] == (x, y):
                    data[key] = None
                    updated = True
        if updated:
            self.update_agent_info()

    def assign_agent_position(self, agent_id: int, role: str, x: int, y: int) -> None:
        data = self.agent_data[agent_id]
        old_pos = data[role]
        if old_pos == (x, y):
            return
        if old_pos:
            self.update_cell_display(old_pos[0], old_pos[1])
        for other_id, other_data in self.agent_data.items():
            if other_id == agent_id:
                continue
            if other_data[role] == (x, y):
                other_data[role] = None
                if other_id == self.selected_agent():
                    self.update_agent_info()
        data[role] = (x, y)
        self.update_agent_info()

    def agent_at(self, role: str, x: int, y: int) -> Optional[int]:
        for agent_id, data in self.agent_data.items():
            if data[role] == (x, y):
                return agent_id
        return None

    def update_cell_display(self, x: int, y: int) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        rect_id = self.rectangles[y][x]
        text_id = self.cell_text[y][x]
        cell_value = self.grid[y][x]
        start_agent = self.agent_at("start", x, y)
        goal_agent = self.agent_at("goal", x, y)

        if cell_value == OBSTACLE_CELL:
            fill = "#2f3542"
            text = ""
            text_color = "#f1f2f6"
        else:
            if start_agent is not None and goal_agent is not None:
                fill = "#8e44ad"
                if start_agent == goal_agent:
                    text = f"{start_agent}*"
                else:
                    text = f"S{start_agent}/G{goal_agent}"
                text_color = "#f1f2f6"
            elif start_agent is not None:
                fill = "#3f8efc"
                text = f"S{start_agent}"
                text_color = "#f1f2f6"
            elif goal_agent is not None:
                fill = "#2ecc71"
                text = f"G{goal_agent}"
                text_color = "#f1f2f6"
            else:
                fill = "#ffffff"
                text = ""
                text_color = "#2f3542"

        self.canvas.itemconfigure(rect_id, fill=fill)
        self.canvas.itemconfigure(text_id, text=text, fill=text_color)

    def clear_obstacles(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == OBSTACLE_CELL:
                    self.grid[y][x] = EMPTY_CELL
                    self.update_cell_display(x, y)

    def browse_output_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.output_dir_var.get() or DEFAULT_OUTPUT_DIR)
        if selected:
            self.output_dir_var.set(selected)

    def generate_map(self) -> None:
        map_name = self.map_name_var.get().strip()
        if not map_name:
            messagebox.showerror("Validation error", "Map name cannot be empty.")
            return

        try:
            agents_min = int(self.agent_min_var.get())
            agents_max = int(self.agent_max_var.get())
        except ValueError:
            messagebox.showerror("Validation error", "Agent min/max must be integers.")
            return

        if agents_min < 0 or agents_max < 0:
            messagebox.showerror("Validation error", "Agent min/max cannot be negative.")
            return
        if agents_min > agents_max:
            messagebox.showerror("Validation error", "Agent min cannot exceed agent max.")
            return

        incomplete_agents = [
            agent
            for agent, data in self.agent_data.items()
            if data["start"] is None or data["goal"] is None
        ]
        if incomplete_agents:
            pretty_agents = ", ".join(str(agent) for agent in sorted(incomplete_agents))
            proceed = messagebox.askyesno(
                "Incomplete agents",
                "Some agents are missing start or goal positions and will be excluded "
                f"from the saved file.\nAgents: {pretty_agents}\nContinue?",
            )
            if not proceed:
                return

        complete_agents = {
            agent: data
            for agent, data in self.agent_data.items()
            if data["start"] is not None and data["goal"] is not None
        }

        start_positions = {agent: data["start"] for agent, data in complete_agents.items()}
        goal_positions = {agent: data["goal"] for agent, data in complete_agents.items()}
        augmentations = [name for name, var in self.augment_vars.items() if var.get()]
        output_dir = self.output_dir_var.get().strip() or DEFAULT_OUTPUT_DIR

        try:
            output_path = generate_json(
                map_name,
                agents_min,
                agents_max,
                [row[:] for row in self.grid],
                start_positions,
                goal_positions,
                augmentations,
                output_dir=output_dir,
                overwrite=False,
            )
        except FileExistsError:
            overwrite = messagebox.askyesno(
                "File exists",
                "The target JSON already exists. Overwrite it?\nA .bak backup will be created.",
            )
            if not overwrite:
                return
            try:
                output_path = generate_json(
                    map_name,
                    agents_min,
                    agents_max,
                    [row[:] for row in self.grid],
                    start_positions,
                    goal_positions,
                    augmentations,
                    output_dir=output_dir,
                    overwrite=True,
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                messagebox.showerror("Generation failed", str(exc))
                return
        except Exception as exc:  # pylint: disable=broad-exception-caught
            messagebox.showerror("Generation failed", str(exc))
            return

        if self.preview_var.get():
            try:
                preview_json(output_path)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                messagebox.showwarning("Preview failed", f"Preview could not be displayed:\n{exc}")

        messagebox.showinfo("Success", f"JSON map saved to:\n{output_path}")


def main() -> None:
    root = tk.Tk()
    app = MapEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
