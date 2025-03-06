import numpy as np

class IonCollector:
    def __init__(self, R, phi):
        """
        Initialize the IonCollector class with radius and angle.
        
        Parameters:
        R (float): The radius of the cylindrical collector.
        phi (float): The angle that divides heavy and light ions.
        """
        self.R = R
        self.phi = phi
        self.collected_ions = []

    def collect_ions(self, ion_trajectory, region):
        """
        Collect ions that pass through a specified region.
        
        Parameters:
        ion_trajectory (IonTrajectory): The ion trajectory data.
        region (tuple): The region defined by (x_min, x_max, y_min, y_max, z_min, z_max).
        """
        x_min, x_max, y_min, y_max, z_min, z_max = region
        positions = ion_trajectory.get_positions()
        mask = (
            (positions[:, 0] >= x_min) & (positions[:, 0] <= x_max) &
            (positions[:, 1] >= y_min) & (positions[:, 1] <= y_max) &
            (positions[:, 2] >= z_min) & (positions[:, 2] <= z_max)
        )
        self.collected_ions.append(positions[mask])

    def get_collected_ions(self):
        """
        Get the collected ions.
        
        Returns:
        np.array: The positions of the collected ions.
        """
        return np.concatenate(self.collected_ions, axis=0) if self.collected_ions else np.array([])
