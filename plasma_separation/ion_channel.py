import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from plasma_separation.ion_trajectory import IonTrajectories
import matplotlib.pyplot as plt  # Add this import for plotting


class IonChannel:
    def __init__(
        self,
        ion_trajectory: IonTrajectories,
        width = 5.0,
        ionizer_R = 15.0,
        indent = 4.0,
        channel_len = 10.0
    ):
        """
        Initialize the IonChannel class with width, ionizer radius, ion trajectory data.

        Parameters:
        ion_trajectory (IonTrajectories): The ions trajectory data.
        width (float): The width of the channel in cm.
        ionizer_R (float): The radius of cylindrical ionizer in cm.
        """
        self.width = width
        self.ionizer_R = ionizer_R
        self.indent = indent
        self.len = channel_len
        self.ion_trajectory = ion_trajectory
        self.ion_trajectory.poses_cyl = IonTrajectories.cartesian_to_cylindrical(ion_trajectory.poses)
        # Initialize as empty numpy arrays
        self.collected_ions_pos = np.empty(
            (0, 2)
        )  # (N_ions, 2) array of phi, z coordinates of collected ions. If ion is not collected, the value is None.
        self.ions_pass_flag = np.empty(0)  # (N_ions) array of booleans (1 if ion entered the channel, 0 if not)
        self.stayed_flag = np.empty(0) # (N_ions) array of booleans (1 if ion stayed in ionizer till end, 0 if not)
        # Collect ions that entered the channel or deposited on its walls
        self.collect_ions()

    def collect_ions(self):
        """
        Collect ions that pass through the x = ionizer_R surface during their first pass and that cross 
        abs(y) = width /2 surface.

        Parameters:
        ion_trajectory (IonTrajectories): The ions trajectory data.
        """
        r_ionizer = self.ionizer_R
        channel_len = self.len
        n_ions = self.ion_trajectory.poses.shape[0]
        self.collected_ions_pos = np.empty((n_ions, 3))
        self.ions_pass_flag = np.empty(n_ions)
        self.stayed_flag = np.empty(n_ions)

        for i in range(n_ions):
            ion_positions = self.ion_trajectory.poses[i]
            x = ion_positions[:, 0]
            y = ion_positions[:, 1]
            z = ion_positions[:, 2]
            r = self.ion_trajectory.poses_cyl[i][:,0]
           
            first_wall_pass_index = 0
            while first_wall_pass_index < len(x) and (
                r[first_wall_pass_index] < r_ionizer
                or abs(y[first_wall_pass_index]) < self.width /2
                or x[first_wall_pass_index] < (r_ionizer**2 - 0 * (self.width/2)**2)**0.5
                or x[first_wall_pass_index] > r_ionizer + channel_len
            ):
                first_wall_pass_index += 1
 
            if first_wall_pass_index < len(x):
                x1 = x[first_wall_pass_index]
                y1 = y[first_wall_pass_index]
                z1 = z[first_wall_pass_index]

                self.collected_ions_pos[i] = [x1, y1, z1]

            else:
                self.collected_ions_pos[i] = [None, None, None]
        
        x_dep = self.collected_ions_pos[:,0]
        self.deposition_len = np.max(x_dep[~np.isnan(x_dep)])

        for i in range(n_ions):
            
            ion_positions = self.ion_trajectory.poses[i]
            x = ion_positions[:, 0]
            y = ion_positions[:, 1]
            z = ion_positions[:, 2]
            r = self.ion_trajectory.poses_cyl[i][:,0]

            if x[-1] > self.deposition_len:
                self.ions_pass_flag[i] = 1
            else:
                self.ions_pass_flag[i] = 0
            if r[-1] <r_ionizer:
                self.stayed_flag[i] = 1
            else:
                self.stayed_flag[i] = 0
        return self.stayed_flag, self.ions_pass_flag, self.collected_ions_pos


    def plot_collected_ions(self):
        """
        Plot collected ions in (x, y) coordinates. 
        Add histogram with x ions deposition.
        """
        x = self.collected_ions_pos[:, 0]
        y = self.collected_ions_pos[:, 1]

        fig, ax = plt.subplots(
            2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1]}, layout="constrained"
        )

        # Scatter plot
        scatter = ax[0].scatter(x, y, marker="o", alpha=0.5)
        #ax[0].set_title(f"Collected Ions (R={self.R} cm)")
        ax[0].set_xlabel("x, см")
        ax[0].set_ylabel("y, см")


        # Histogram
        ax[1].hist(
            x,
            bins=30,
            alpha=0.5,
            color="blue",
            orientation="vertical",
        )
        ax[1].set_ylabel("Count")

        # Share x-axis
        ax[1].sharex(ax[0])

        return fig
