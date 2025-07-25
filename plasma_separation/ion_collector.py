import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import bisect
from plasma_separation.ion_trajectory import IonTrajectories
import matplotlib.pyplot as plt  # Add this import for plotting


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
        mass_threshold=175,
        R=20,
        phi=np.pi / 2,
        z_min=-np.inf,
        phi_min=0,
    ):
        """
        Initialize the IonCollector class with radius, angle, region limits, ion trajectory data, and mass threshold.

        Parameters:
        ion_trajectory (IonTrajectories): The ions trajectory data.
        mass_threshold (float): The mass threshold to classify ions as light or heavy in u.
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
        self.ion_trajectory.poses_cyl = IonTrajectories.cartesian_to_cylindrical(ion_trajectory.poses)
        self.mass_threshold = mass_threshold
        # Initialize as empty numpy arrays
        self.collected_ions_pos = np.empty(
            (0, 2)
        )  # (N_ions, 2) array of phi, z coordinates of collected ions. If ion is not collected, the value is None.
        self.collected_ions_mass = np.empty(0)  # (N_ions) array of collected ions mass

        # Collect ions and calculate mass classes
        self.collect_ions()
        self.mass_classes = np.where(
            self.collected_ions_mass < self.mass_threshold, 1, 0
        )  # Light ions: 1, Heavy ions: 0

    def collect_ions(self):
        """
        Collect ions that pass through the r=R surface during their first pass and within the specified region.

        Parameters:
        ion_trajectory (IonTrajectories): The ions trajectory data.
        """
        r_collector = self.R
        n_ions = self.ion_trajectory.poses_cyl.shape[0]
        self.collected_ions_pos = np.empty((n_ions, 2))
        self.collected_ions_mass = np.empty(n_ions)

        for i in range(n_ions):
            ion_positions = self.ion_trajectory.poses_cyl[i]
            ion_mass = self.ion_trajectory.ions_mass[i]
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
                    r1, phi1, z1  = ion_positions[first_pass_index - 1]
                    r2, phi2, z2  = ion_positions[first_pass_index]
                    z_c = np.interp(r_collector, [r1, r2], [z1, z2])
                    phi_c = np.interp(r_collector, [r1, r2], [phi1, phi2])
                else:
                    print(i)
                    print(first_pass_index)
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
            np.isnan(self.collected_ions_pos[:, 0]),
            2,  # Uncollected ions
            self.collected_ions_pos[:, 0]
            < self.phi,  # Light ions: True, Heavy ions: False
        )

    def quality_smirnov(self):
        '''
        Calculates metrics from
        V. S. Smirnov, R. O. Egorov, S. A. Kislenko, N. N. Antonov, V. P. Smirnov, A. V. Gavrikov; 
        Simulation of ion flux of actinides and uranium fission products in the plasma separator with a potential well.
        Phys. Plasmas 1 November 2020; 27 (11): 113503. https://doi.org/10.1063/5.0020001
        '''
    
        actinides_phis = self.collected_ions_pos[self.collected_ions_mass >= 235, 0]
        actinides_center_phi = np.arctan2(np.mean(np.sin(actinides_phis)), np.mean(np.cos(actinides_phis)))
    
        heavy_phis = self.collected_ions_pos[(self.collected_ions_mass >= 120) * (self.collected_ions_mass <= 160), 0]
        heavy_center_phi = np.arctan2(np.mean(np.sin(heavy_phis)), np.mean(np.cos(heavy_phis)))
    
        L = self.R * (actinides_center_phi - heavy_center_phi) 
    
        percent = 90
        d_actinides =  2.0 * self.R *  np.percentile(np.abs(actinides_phis - actinides_center_phi), percent, method = 'inverted_cdf') 
        d_heavy =  2.0 * self.R * np.percentile(np.abs(heavy_phis - heavy_center_phi), percent, method = 'inverted_cdf') 
        
        quality = L / (d_actinides + d_heavy) * 2.0
        
        return quality, L, d_actinides, d_heavy
   
    def quality_purification_degree(self, eps = 1e-3):
        '''
        Calculates fraction of heavy particles in a spot of light particles and vice versa.
        For this purpose phi is optimised.

        Parameters:
        eps (float): Step size used to calculate gradient in phi optimization procedure.
        '''
        actinides_phis = self.collected_ions_pos[self.collected_ions_mass >= 235, 0]
        actinides_center_phi = np.arctan2(np.mean(np.sin(actinides_phis)), np.mean(np.cos(actinides_phis)))
        heavy_phis = self.collected_ions_pos[(self.collected_ions_mass >= 120) * (self.collected_ions_mass <= 160), 0]
        heavy_center_phi = np.arctan2(np.mean(np.sin(heavy_phis)), np.mean(np.cos(heavy_phis)))
        
        percent = 99
        r_heavy = np.percentile(np.abs(heavy_phis - heavy_center_phi), percent, method = 'inverted_cdf') 
        
        phi_0 = heavy_center_phi

        def loss_function(classes_true, classes_pred, phi):
            N_heavy_in_actinides = np.sum(classes_true * (~classes_pred))
            N_actinides_in_actinides = np.sum((~classes_true) * (~classes_pred))
            N_actinides_in_heavy = np.sum((~classes_true) * classes_pred)
            N_heavy_in_heavy = np.sum(classes_true * classes_pred)

            frac_heavy_in_actinides = N_heavy_in_actinides / N_actinides_in_actinides
            frac_actinides_in_heavy = N_actinides_in_heavy / N_heavy_in_heavy

            loss = max(frac_heavy_in_actinides, frac_actinides_in_heavy)
            #loss = abs(N_heavy_in_actinides - N_actinides_in_heavy)
            return loss
        
        self.phi = self.optimize_phi(loss_function, phi_0, phi_min = heavy_center_phi, phi_max = actinides_center_phi, eps = eps) 
        #self.phi = bisect(objective, heavy_center_phi, actinides_center_phi) # this optimization method may be better

        classes_true = self.mass_classes.astype(bool)
        classes_pred = self.identify_classes().astype(bool)

        N_heavy_in_actinides = np.sum(classes_true * (~classes_pred))
        N_actinides_in_actinides = np.sum((~classes_true) * (~classes_pred))
        N_actinides_in_heavy = np.sum((~classes_true) * classes_pred)
        N_heavy_in_heavy = np.sum(classes_true * classes_pred)

        frac_heavy_in_actinides = N_heavy_in_actinides / N_actinides_in_actinides
        frac_actinides_in_heavy = N_actinides_in_heavy / N_heavy_in_heavy
        
        return frac_heavy_in_actinides, frac_actinides_in_heavy
        
    def optimize_phi(
        self,
        loss_function,
        phi_0=np.pi / 2,
        phi_min=0,
        phi_max=2 * np.pi,
        eps = 1e-3
    ):
        """
        Optimize the phi parameter to minimize the loss function for ion classification.

        Parameters:
        loss_function (callable): The loss function to minimize.
        phi_0 (float): Initial guess for phi.
        phi_min (float): Minimum value for phi.
        phi_max (float): Maximum value for phi.
        eps (float): Step size for gradient calculation in minimization algorithm.
        """

        # TODO: filtering should be corrected, as it produces errors
        # Filter out None values from collected_ions_pos
        #valid_indices = np.where(self.collected_ions_pos[:, 0] is not None)
        #filtered_collected_ions_pos = self.collected_ions_pos[valid_indices]

        def objective(phi):
            self.phi = phi
            y_true = self.mass_classes.astype(bool)
            y_pred = self.identify_classes().astype(bool)
            return loss_function(y_true, y_pred, phi)

        result = minimize(objective, phi_0, bounds=[(phi_min, phi_max)], options={'disp': True, 'ftol': 1e-8, 'eps': eps}, method = "SLSQP")
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

    def plot_collected_ions(self):
        """
        Plot collected ions in (z, R*phi) coordinates. Color dots according to ion mass using rainbow colormap.
        Include collector R in title. Plot phi_threshold as a black -- line. Add histogram with R*phi ions deposition.
        """
        z = self.collected_ions_pos[:, 1]
        R_phi = self.R * self.collected_ions_pos[:, 0]
        masses = self.collected_ions_mass

        fig, ax = plt.subplots(
            1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 1]}, layout="constrained"
        )

        # Scatter plot
        scatter = ax[0].scatter(z, R_phi, c=masses, cmap="rainbow", marker="o", alpha=0.5)
        #ax[0].set_title(f"Collected Ions (R={self.R} cm)")
        ax[0].set_xlabel("z, см")
        ax[0].set_ylabel("R*phi, см")
        plt.colorbar(scatter, ax=ax[0], label="Масса иона, а. е. м.")

        # Plot phi_threshold line
        phi_threshold = self.phi
        
        ax[0].axhline(
            y=self.R * phi_threshold,
            color="black",
            linestyle="--",
            label="phi_threshold",
        )
        ax[0].legend()
        


        # Histogram
        light_ions = R_phi[self.mass_classes == 1]
        heavy_ions = R_phi[self.mass_classes == 0]
        ax[1].hist(
            light_ions,
            bins=30,
            alpha=0.5,
            label="Light Ions",
            color="blue",
            orientation="horizontal",
        )
        ax[1].hist(
            heavy_ions,
            bins=30,
            alpha=0.5,
            label="Heavy Ions",
            color="red",
            orientation="horizontal",
        )
        ax[1].set_xlabel("Count")
        ax[1].legend()

        # Share y-axis
        ax[1].sharey(ax[0])

        return fig
