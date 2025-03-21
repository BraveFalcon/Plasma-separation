"""
This module provides functionality to handle ion trajectories, including importing data from COMSOL and OpenMM files,
converting coordinates, plotting trajectories and velocity distributions, and saving/loading trajectory data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pandas as pd
import scipy.constants
import os
import sys
import h5py


_load_save_filename = "trajectories.npz"


def load_from_dir(dirpath, remove_original=False):
    """
    Load ion trajectories from a directory. If a saved file exists, load from it. Otherwise, import from COMSOL or OpenMM files.

    Parameters:
    dirpath (str): The path to the directory containing the ion trajectory data.
    remove_original (bool): Whether to remove the original files after loading. Default is False.

    Returns:
    IonTrajectories: An instance of the IonTrajectories class.
    """
    os.chdir(dirpath)
    paths = os.listdir(dirpath)

    comsol_filename = "ion_trajectories_comsol.csv"
    openmm_filename = "ion_trajectories_openmm.h5"

    if _load_save_filename in paths:
        return IonTrajectories.load(dirpath)

    if comsol_filename in paths:
        traj = import_from_comsol_file(os.path.join(dirpath, comsol_filename))
        traj.save(dirpath)
        if remove_original:
            os.remove(os.path.join(dirpath, comsol_filename))
    elif openmm_filename in paths:
        traj = import_from_openmm_file(os.path.join(dirpath, openmm_filename))
        traj.save(dirpath)
        if remove_original:
            os.remove(os.path.join(dirpath, openmm_filename))
    else:
        raise Exception(f"Can't find trajectories file in the directory {dirpath}.")

    return traj


def import_from_comsol_file(filepath):
    """
    Import ion trajectories data from a COMSOL-generated CSV file and create an IonTrajectories instance.

    Parameters:
    filepath (str): The path to the CSV file containing the ion trajectories data.

    Returns:
    IonTrajectories: An instance of the IonTrajectories class.
    """
    df = pd.read_csv(filepath, skiprows=7)
    df = df.rename(columns={"% Index": "Index"})
    ion_indices = df["Index"].unique()
    N_ions = len(ion_indices)
    N_frames = df["t"].nunique()

    poses = np.zeros((N_ions, N_frames, 3))
    vels = np.zeros((N_ions, N_frames, 3))
    ts = np.sort(df["t"].unique()) * 1e6  # Convert seconds to microseconds
    ions_mass = np.zeros(N_ions)

    valid_indices = []
    for i, index in enumerate(ion_indices):
        ion_data = df[df["Index"] == index]
        if ion_data[["qx (cm)", "qy (cm)", "qz (cm)"]].isnull().values.any():
            continue  # Skip ions with NaN positions
        valid_indices.append(i)
        ions_mass[i] = ion_data["cpt.mp (u)"].values[0]
        poses[i, :, 0] = ion_data["qx (cm)"].values
        poses[i, :, 1] = ion_data["qy (cm)"].values
        poses[i, :, 2] = ion_data["qz (cm)"].values
        vels[i, :, 0] = ion_data["cpt.vx (m/s)"].values
        vels[i, :, 1] = ion_data["cpt.vy (m/s)"].values
        vels[i, :, 2] = ion_data["cpt.vz (m/s)"].values

    poses = poses[valid_indices]
    vels = vels[valid_indices]
    ions_mass = ions_mass[valid_indices]
    return IonTrajectories(poses, vels, ts, ions_mass)


def import_from_openmm_file(filepath):
    """
    Import ion trajectories data from a OPENMM-generated ??? file and create an IonTrajectories instance.

    Parameters:
    filepath (str): The path to the ??? file containing the ion trajectories data.

    Returns:
    IonTrajectories: An instance of the IonTrajectories class.
    """
    ions_mass = []
    poses = []
    vels = []
    ts = []

    with h5py.File(filepath, "r") as f:
        # load ion mases from topology
        topology = np.array(f["topology"])[0].decode("UTF-8").split()
        N_atoms = np.array(f["coordinates"]).shape[1]
        for i in range(N_atoms):
            ions_mass.append(int(topology[10 + i * 6][1:-2]))
        ions_mass = np.array(ions_mass)
        # load ions poses
        poses = np.array(f["coordinates"])
        poses = np.moveaxis(
            poses, (0, 1, 2), (1, 0, 2)
        )  # convert shape from (N_frames, N_ions, 3) to (N_ions, N_frames, 3)
        poses /= 1e7  # conversion from nm to cm

        # load ions vels
        vels = np.array(f["velocities"]) * 1e-9 / 1e-12
        vels = np.moveaxis(vels, (0, 1, 2), (0, 2, 1))
        vels = np.moveaxis(
            vels, (0, 1, 2), (1, 2, 0)
        )  # convert shape to (N_ions, N_frames, 3)

        ts = np.array(f["time"]) / 1e6  # convert ps to micro seconds
    return IonTrajectories(poses, vels, ts, ions_mass)


class IonTrajectories:

    def __init__(self, poses, vels, ts, ions_mass):
        """
        Initialize the IonTrajectories class with positions, velocities, time steps, and ion masses.

        Parameters:
        poses (np.array): Array of positions of the ions in Cartesian coordinates (N_ions, N_frames, 3) in cm.
        vels (np.array): Array of velocities of the ions in Cartesian coordinates (N_ions, N_frames, 3) in m/s.
        ts (np.array): Array of time steps (N_frames) in μs.
        ions_mass (np.array): Array of ion masses (N_ions) in unified atomic mass unit (u).
        """
        self.poses = np.array(poses)
        self.vels = np.array(vels)
        self.ts = np.array(ts)
        self.ions_mass = np.array(ions_mass)

    def save(self, dirpath):
        """
        Save the ion trajectories data to a compressed file.

        Parameters:
        dirpath (str): The path to the directory where the data should be saved.
        """
        np.savez_compressed(
            os.path.join(dirpath, _load_save_filename),
            poses=self.poses,
            vels=self.vels,
            ts=self.ts,
            ions_mass=self.ions_mass,
        )

    @classmethod
    def load(cls, dirpath):
        """
        Load the ion trajectories data from a compressed file.

        Parameters:
        dirpath (str): The path to the directory containing the saved data.

        Returns:
        IonTrajectories: An instance of the IonTrajectories class.
        """
        return cls(**np.load(os.path.join(dirpath, _load_save_filename)))

    @staticmethod
    def cartesian_to_cylindrical(cartesian_coords):
        """
        Convert Cartesian coordinates to cylindrical coordinates.

        Parameters:
        cartesian_coords (np.array): Array of Cartesian coordinates (x, y, z).

        Returns:
        np.array: Array of cylindrical coordinates (r, theta, z).
        """
        x, y, z = (
            cartesian_coords[:, :, 0],
            cartesian_coords[:, :, 1],
            cartesian_coords[:, :, 2],
        )
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return np.stack((r, phi, z), axis=-1)

    @staticmethod
    def cylindrical_to_cartesian(cylindrical_coords):
        """
        Convert cylindrical coordinates to Cartesian coordinates.

        Parameters:
        cylindrical_coords (np.array): Array of cylindrical coordinates (r, theta, z).

        Returns:
        np.array: Array of Cartesian coordinates (x, y, z).
        """
        r, phi, z = (
            cylindrical_coords[:, :, 0],
            cylindrical_coords[:, :, 1],
            cylindrical_coords[:, :, 2],
        )
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return np.stack((x, y, z), axis=-1)

    def plot_initial_velocity_distribution(self, ax):
        """
        Plot the initial velocity norm distribution of the ions.

        Parameters:
        ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axes object.
        """
        initial_velocities = self.vels[:, 0, :]
        velocity_norms = np.linalg.norm(initial_velocities, axis=1)
        kin_energies = (
            0.5
            * self.ions_mass
            * scipy.constants.atomic_mass
            * velocity_norms**2
            / scipy.constants.e
        )
        ax.hist(kin_energies, bins=10, color="blue", alpha=0.7)
        ax.set_xlabel("Kinetic Energy (eV)")
        ax.set_ylabel("Frequency")
        ax.set_title("Initial Kinetic Energy Distribution")

    def plot_initial_velocity_angle_distribution(self, ax):
        """
        Plot the distribution of the angle between the initial ion velocities and the z-axis.

        Parameters:
        ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axes object.
        """
        initial_velocities = self.vels[:, 0, :]
        velocity_norms = np.linalg.norm(initial_velocities, axis=1)
        angles = np.arccos(
            initial_velocities[:, 2] / velocity_norms
        )  # Calculate angle with z-axis
        ax.hist(np.degrees(angles), bins=15, color="green", alpha=0.7)
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("Frequency")
        ax.set_title("Initial Velocity Angle Distribution")

    def plot_trajectories_cartesian(self, ax, cmap):
        """
        Plot ion trajectories in 3D Cartesian coordinates.

        Parameters:
        ax (matplotlib.axes._subplots.Axes3DSubplot): Matplotlib 3D axes object.
        cmap (matplotlib.colors.Colormap): Matplotlib colormap object.
        """
        colors = self._get_colors(cmap)
        poses_cartesian = self.poses  # Already in Cartesian coordinates
        for i, color in enumerate(colors):
            ax.plot(
                poses_cartesian[i, :, 0],
                poses_cartesian[i, :, 1],
                poses_cartesian[i, :, 2],
                color=color,
                alpha=0.25,
            )
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_zlabel("z (cm)")
        ax.set_title("Ion Trajectories in 3D Cartesian Coordinates")

    def plot_trajectories_xy(self, ax, cmap):
        """
        Plot ion trajectories in the XY plane.

        Parameters:
        ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axes object.
        cmap (matplotlib.colors.Colormap): Matplotlib colormap object.
        """
        colors = self._get_colors(cmap)
        poses_cartesian = self.poses  # Already in Cartesian coordinates
        for i, color in enumerate(colors):
            ax.plot(
                poses_cartesian[i, :, 0],
                poses_cartesian[i, :, 1],
                color=color,
                alpha=0.25,
            )
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title("Ion Trajectories in XY Coordinates")

    def plot_trajectories_zphi(self, ax, cmap):
        """
        Plot ion trajectories in (z, phi) coordinates.

        Parameters:
        ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axes object.
        cmap (matplotlib.colors.Colormap): Matplotlib colormap object.
        """
        colors = self._get_colors(cmap)
        for i, color in enumerate(colors):
            z = self.poses[i, :, 2]
            phi = np.arctan2(self.poses[i, :, 1], self.poses[i, :, 0])
            ax.plot(z, phi, color=color, alpha=0.25)
        ax.set_xlabel("z (cm)")
        ax.set_ylabel("phi (rad)")
        ax.set_title("Ion Trajectories in ZPhi Coordinates")

    def plot_all(self):
        """
        Plot all ion trajectories and velocity distributions in a single figure with 5 subplots.

        Returns:
        matplotlib.figure.Figure: The created figure.
        """

        layout = [["3D", "XY", "VelNorm"], ["3D", "ZPhi", "VelAngle"]]
        fig, axd = plt.subplot_mosaic(
            layout,
            figsize=np.array((3, 2)) * 4,
            per_subplot_kw={"3D": {"projection": "3d"}},
            layout="constrained",
        )
        cmap = get_cmap("rainbow")

        # 3D Cartesian plot
        self.plot_trajectories_cartesian(axd["3D"], cmap)

        # 2D XY plot
        self.plot_trajectories_xy(axd["XY"], cmap)

        # 2D ZPhi plot
        self.plot_trajectories_zphi(axd["ZPhi"], cmap)

        # Initial velocity norm distribution plot
        self.plot_initial_velocity_distribution(axd["VelNorm"])

        # Initial velocity angle distribution plot
        self.plot_initial_velocity_angle_distribution(axd["VelAngle"])

        # Colorbar
        norm = plt.Normalize(self.ions_mass.min(), self.ions_mass.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=[axd["3D"]],
            orientation="vertical",
            fraction=0.02,
            pad=0.04,
            location="left",
        )
        cbar.set_label("Ion Mass (u)")

        return fig

    def _get_colors(self, cmap):
        """
        Get colors for each ion based on their mass using the provided colormap.

        Parameters:
        cmap (matplotlib.colors.Colormap): Matplotlib colormap object.

        Returns:
        list: List of colors for each ion.
        """
        # TODO: fix. It should one colormap for all.
        norm = plt.Normalize(self.ions_mass.min(), self.ions_mass.max())
        return [cmap(norm(mass)) for mass in self.ions_mass]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("No path specified")
    path = sys.argv[1]
    if os.path.isdir(path):
        traj = load_from_dir(path)
    else:
        raise Exception("Specify directory with a trajectories file")

    fig = traj.plot_all()
    fig.savefig(os.path.join(path, "trajectories.png"), dpi=100)
