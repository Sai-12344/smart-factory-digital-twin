import numpy as np
import pandas as pd
from .machine_physics import MachinePhysics


class FactoryDatasetGenerator:
    """
    Generates complete datasets for:
    - LSTM RUL model
    - Autoencoder anomaly detection
    - RL training state data
    """

    def __init__(self, num_machines=20, cycles_per_machine=300, curve_mode="hybrid"):
        self.num_machines = num_machines
        self.cycles_per_machine = cycles_per_machine
        self.curve_mode = curve_mode

    def generate_machine_data(self, machine_id):
        machine = MachinePhysics(machine_id, curve_mode=self.curve_mode)
        records = []

        for _ in range(self.cycles_per_machine):
            action = "run"  # default action
            record = machine.simulate_step(action=action)
            records.append(record)

        return records

    def generate_full_dataset(self):
        all_data = []

        for m in range(self.num_machines):
            print(f"Simulating machine {m}...")
            machine_data = self.generate_machine_data(m)
            all_data.extend(machine_data)

        df = pd.DataFrame(all_data)

        # Sort for clean data output
        df = df.sort_values(by=["id", "cycle"]).reset_index(drop=True)

        return df

    def save_dataset(self, df, filename="factory_dataset.csv"):
        df.to_csv(filename, index=False)
        print(f"Saved dataset → {filename}")

    def generate_and_save(self, filename="factory_dataset.csv"):
        df = self.generate_full_dataset()
        self.save_dataset(df, filename)
        return df