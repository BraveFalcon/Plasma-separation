import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from ion_trajectory import IonTrajectories


def classification_loss(y_true, y_pred):
    """
    Standard classification loss function with a high penalty for uncollected ions.

    Parameters:
    y_true (np.ndarray): True labels (0 for heavy ions, 1 for light ions, 2 for uncollected ions).
    y_pred (np.ndarray): Predicted labels (0 for heavy ions, 1 for light ions, 2 for uncollected ions).

    Returns:
    float: The computed loss value.
    """
    penalty_uncollected = 1000  # High penalty for uncollected ions
    loss = np.sum(
        (y_true != y_pred) * (y_true != 2) + (y_true == 2) * penalty_uncollected
    )
    return loss


class IonCollector:
    def __init__(
        self,
        ion_trajectory: IonTrajectories,
        mass_threshold,
        R=20,
        phi=np.pi / 2,
        z_min=-np.inf,
        phi_min=0,
    ):
        """
        Initialize the IonCollector class with radius, angle, region limits, ion trajectory data, and mass threshold.

        Parameters:
        ion_trajectory (IonTrajectories): The ions trajectory data.
        mass_threshold (float): The mass threshold to classify ions as light or heavy.
        R (float): The radius of the cylindrical collector in cm.
        phi (float): The angle that divides heavy and light ions in radians.
        z_min (float): The minimum z-coordinate for the collector region in cm.
        phi_min (float): The minimum phi-coordinate for the collector region in radians.
        """
        self.R = R
        self.phi = phi
        self.z_min = z_min
        self.phi_min = phi_min
        self.ion_trajectory = ion_trajectory
        self.mass_threshold = mass_threshold
        # Initialize as empty numpy arrays
        self.collected_ions_pos = np.empty(
            (0, 2)
        )  # (N_ions, 2) array of phi, z coordinates of collected ions. If ion is not collected, the value is None.
        self.collected_ions_mass = np.empty(0)  # (N_ions) array of collected ions mass

        # Collect ions and calculate mass classes
        self.collect_ions(ion_trajectory)
        self.mass_classes = np.where(
            self.collected_ions_mass < self.mass_threshold, 1, 0
        )  # Light ions: 1, Heavy ions: 0

    def collect_ions(self, ion_trajectory: IonTrajectories):
        """
        Collect ions that pass through the r=R surface during their first pass and within the specified region.

        Parameters:
        ion_trajectory (IonTrajectories): The ions trajectory data.
        """
        self.ion_trajectory = ion_trajectory
        r_collector = self.R
        n_ions = ion_trajectory.poses.shape[0]
        self.collected_ions_pos = np.empty((n_ions, 2))
        self.collected_ions_mass = np.empty(n_ions)

        for i in range(n_ions):
            ion_positions = ion_trajectory.poses[i]
            ion_mass = ion_trajectory.ions_mass[i]
            r = ion_positions[:, 0]
            phi = ion_positions[:, 1]
            z = ion_positions[:, 2]
            first_pass_index = 0

            while first_pass_index < len(r) and (
                r[first_pass_index] < r_collector
                or z[first_pass_index] < self.z_min
                or phi[first_pass_index] < self.phi_min
            ):
                first_pass_index += 1

            if first_pass_index < len(r):
                if first_pass_index > 0:
                    r1, z1, phi1 = ion_positions[first_pass_index - 1]
                    r2, z2, phi2 = ion_positions[first_pass_index]
                    z_c = np.interp(r_collector, [r1, r2], [z1, z2])
                    phi_c = np.interp(r_collector, [r1, r2], [phi1, phi2])
                else:
                    raise Exception("Collector is too close to the injection point.")

                self.collected_ions_pos[i] = [phi_c, z_c]

            else:
                self.collected_ions_pos[i] = [None, None]

            self.collected_ions_mass[i] = ion_mass

        return self.collected_ions_pos, self.collected_ions_mass

    def identify_classes(self):
        """
        Identify the classes of ions based on their mass and collection status.

        Returns:
        np.ndarray: Array of class labels (0 for heavy ions, 1 for light ions, 2 for uncollected ions).
        """
        return np.where(
            self.collected_ions_pos[:, 0] is None,
            2,  # Uncollected ions
            self.collected_ions_pos[:, 0]
            > self.phi,  # Light ions: True, Heavy ions: False
        )

    def optimize_phi(
        self,
        loss_function,
        phi_0=np.pi / 2,
        phi_min=0,
        phi_max=2 * np.pi,
    ):
        """
        Optimize the phi parameter to minimize the loss function for ion classification.

        Parameters:
        loss_function (callable): The loss function to minimize.
        phi_0 (float): Initial guess for phi.
        phi_min (float): Minimum value for phi.
        phi_max (float): Maximum value for phi.
        """
        # Filter out None values from collected_ions_pos
        valid_indices = np.where(self.collected_ions_pos[:, 0] is not None)
        filtered_collected_ions_pos = self.collected_ions_pos[valid_indices]

        def objective(phi):
            y_true = self.mass_classes
            y_pred = self.identify_classes()
            return loss_function(y_true, y_pred)

        result = minimize(objective, phi_0, bounds=[(phi_min, phi_max)])
        self.phi = result.x[0]
        return self.phi

    def optimize_R(
        self,
        loss_function,
        R_0=20,
        R_min=0,
        R_max=100,
    ):
        """
        Optimize the R parameter to minimize the loss function for ion classification.

        Parameters:
        loss_function (callable): The loss function to minimize.
        R_0 (float): Initial guess for R.
        R_min (float): Minimum value for R.
        R_max (float): Maximum value for R.
        """

        def objective(R):
            self.R = R
            self.collect_ions(self.ion_trajectory)
            y_true = self.mass_classes
            y_pred = self.identify_classes()
            return loss_function(y_true, y_pred)

        result = minimize(objective, R_0, bounds=[(R_min, R_max)])
        self.R = result.x[0]
        return self.R

    def optimize_R_and_phi(
        self,
        loss_function,
        phi_0=np.pi / 2,
        phi_min=0,
        phi_max=2 * np.pi,
        R_0=20,
        R_min=0,
        R_max=100,
        tol=1e-6,
        max_iter=100,
    ):
        """
        Optimize both R and phi parameters to minimize the loss function for ion classification.

        Parameters:
        loss_function (callable): The loss function to minimize.
        phi_0 (float): Initial guess for phi.
        phi_min (float): Minimum value for phi.
        phi_max (float): Maximum value for phi.
        R_0 (float): Initial guess for R.
        R_min (float): Minimum value for R.
        R_max (float): Maximum value for R.
        tol (float): Relative tolerance for stopping criterion.
        max_iter (int): Maximum number of iterations.
        """
        self.phi = phi_0
        self.R = R_0
        prev_loss = float("inf")
        for _ in range(max_iter):
            # Optimize phi
            self.optimize_phi(
                loss_function,
                phi_0=self.phi,
                phi_min=phi_min,
                phi_max=phi_max,
            )
            # Optimize R
            self.optimize_R(
                loss_function,
                R_0=self.R,
                R_min=R_min,
                R_max=R_max,
            )
            # Calculate current loss
            self.collect_ions(self.ion_trajectory)
            y_true = self.mass_classes
            y_pred = self.identify_classes()
            current_loss = loss_function(y_true, y_pred)
            # Check for convergence
            if abs(prev_loss - current_loss) / prev_loss < tol:
                break
            prev_loss = current_loss
        return self.R, self.phi
