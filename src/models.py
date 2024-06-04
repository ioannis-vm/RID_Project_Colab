"""
RID Project
RID|PID models

This file is part of a larger repository aimed at developing a robust
model to generate RID realizations.
https://github.com/ioannis-vm/2023-11_RID_Realizations

"""

# pylint:disable=no-name-in-module


from typing import Optional
from typing import Literal
from typing import Any
import numpy as np
import numpy.typing as npt
import scipy as sp
from scipy.special import erfc
from scipy.special import erfcinv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(formatter={'float': '{:0.5f}'.format})


class Model:
    """
    Base Model class.
    """

    def __init__(self) -> None:
        self.raw_pid: Optional[npt.NDArray] = None
        self.raw_rid: Optional[npt.NDArray] = None

        self.uniform_sample: Optional[npt.NDArray] = None
        self.sim_pid: Optional[npt.NDArray] = None
        self.sim_rid: Optional[npt.NDArray] = None

        self.censoring_limit: Optional[float] = None
        self.parameters: Optional[npt.NDArray] = None
        self.parameter_names: Optional[list[str]] = None
        self.parameter_bounds: Optional[list[tuple[float, float]]] = None
        self.fit_status = False
        self.fit_meta: Any = None

        self.rolling_pid: Optional[npt.NDArray] = None
        self.rolling_rid_50: Optional[npt.NDArray] = None
        self.rolling_rid_20: Optional[npt.NDArray] = None
        self.rolling_rid_80: Optional[npt.NDArray] = None

    def add_data(self, raw_pid: npt.NDArray, raw_rid: npt.NDArray) -> None:
        self.raw_pid = raw_pid
        self.raw_rid = raw_rid

    def calculate_rolling_quantiles(self) -> None:
        # calculate rolling empirical RID|PID quantiles
        assert self.raw_pid is not None
        assert self.raw_rid is not None
        idsort = np.argsort(self.raw_pid)
        num_vals = len(self.raw_pid)
        assert len(self.raw_rid) == num_vals
        rid_vals_sorted = self.raw_rid[idsort]
        pid_vals_sorted = self.raw_pid[idsort]
        group_size = int(len(self.raw_rid) * 0.10)
        rolling_pid = np.array(
            [
                np.mean(pid_vals_sorted[i : i + group_size])
                for i in range(num_vals - group_size + 1)
            ]
        )
        rolling_rid_50 = np.array(
            [
                np.quantile(rid_vals_sorted[i : i + group_size], 0.50)
                for i in range(num_vals - group_size + 1)
            ]
        )
        rolling_rid_20 = np.array(
            [
                np.quantile(rid_vals_sorted[i : i + group_size], 0.20)
                for i in range(num_vals - group_size + 1)
            ]
        )
        rolling_rid_80 = np.array(
            [
                np.quantile(rid_vals_sorted[i : i + group_size], 0.80)
                for i in range(num_vals - group_size + 1)
            ]
        )

        self.rolling_pid = rolling_pid
        self.rolling_rid_50 = rolling_rid_50
        self.rolling_rid_20 = rolling_rid_20
        self.rolling_rid_80 = rolling_rid_80

    def fit(
        self, *args, method: Literal['mle', 'quantiles'] = 'mle', **kwargs
    ) -> None:
        # Initial values

        if method == 'quantiles':
            use_method = self.get_quantile_objective
        elif method == 'mle':
            use_method = self.get_mle_objective

        result = minimize(
            use_method,
            self.parameters,
            bounds=self.parameter_bounds,
            method="Nelder-Mead",
            options={"maxiter": 10000},
            tol=1e-10,
        )

        self.fit_meta = result
        assert result.success, "Minimization failed."
        self.parameters = result.x

    def evaluate_inverse_cdf(self, quantile, pid):
        """
        Evaluate the inverse of the conditional RID|PID CDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_pdf(self, rid, pid, censoring_limit=None):
        """
        Evaluate the conditional RID|PID PDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_cdf(self, rid, pid):
        """
        Evaluate the conditional RID|PID CDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def get_mle_objective(self, parameters: npt.NDArray) -> float:
        # update the parameters
        self.parameters = parameters
        density = self.evaluate_pdf(self.raw_rid, self.raw_pid, self.censoring_limit)
        negloglikelihood = -np.sum(np.log(density))
        return negloglikelihood

    def get_quantile_objective(self, parameters: npt.NDArray) -> float:
        # update the parameters
        self.parameters = parameters

        # calculate the model's RID|PID quantiles
        if self.rolling_pid is None:
            self.calculate_rolling_quantiles()
        model_pid = self.rolling_pid
        model_rid_50 = self.evaluate_inverse_cdf(0.50, model_pid)
        model_rid_20 = self.evaluate_inverse_cdf(0.20, model_pid)
        model_rid_80 = self.evaluate_inverse_cdf(0.80, model_pid)

        loss = (
            (self.rolling_rid_50 - model_rid_50).T
            @ (self.rolling_rid_50 - model_rid_50)
            + (self.rolling_rid_20 - model_rid_20).T
            @ (self.rolling_rid_20 - model_rid_20)
            + (self.rolling_rid_80 - model_rid_80).T
            @ (self.rolling_rid_80 - model_rid_80)
        )

        return loss

    def generate_rid_samples(self, pid_samples: npt.NDArray) -> npt.NDArray:
        if self.uniform_sample is None:
            self.uniform_sample = np.random.uniform(0.00, 1.00, len(pid_samples))

        rid_samples = self.evaluate_inverse_cdf(self.uniform_sample, pid_samples)

        self.sim_pid = pid_samples
        self.sim_rid = rid_samples

        return rid_samples

    def plot_data(self, ax=None, scatter_kwargs=None) -> None:
        """
        Add a scatter plot of the raw data to a matplotlib axis, or
        show it if one is not given.
        """

        if scatter_kwargs is None:
            scatter_kwargs = {
                's': 5.0,
                'facecolor': 'none',
                'edgecolor': 'black',
                'alpha': 0.2,
            }

        if ax is None:
            _, ax = plt.subplots()
            ax.scatter(self.raw_rid, self.raw_pid, **scatter_kwargs)
        if ax is None:
            plt.show()

    def plot_model(
        self, ax, rolling=True, training=True, model=True, model_color='C0'
    ) -> None:
        """
        Plot the data in a scatter plot,
        superimpose their empirical quantiles,
        and the quantiles resulting from the fitted model.
        """

        if self.fit_status == 'False':
            self.fit()

        if training:
            self.plot_data(ax)

        if rolling:
            self.calculate_rolling_quantiles()

            ax.plot(self.rolling_rid_50, self.rolling_pid, 'k')
            ax.plot(self.rolling_rid_20, self.rolling_pid, 'k', linestyle='dashed')
            ax.plot(self.rolling_rid_80, self.rolling_pid, 'k', linestyle='dashed')

        if model:
            # model_pid = np.linspace(0.00, self.rolling_pid[-1], 1000)
            model_pid = np.linspace(0.00, 0.08, 1000)
            model_rid_50 = self.evaluate_inverse_cdf(0.50, model_pid)
            model_rid_20 = self.evaluate_inverse_cdf(0.20, model_pid)
            model_rid_80 = self.evaluate_inverse_cdf(0.80, model_pid)

            ax.plot(model_rid_50, model_pid, model_color)
            ax.plot(model_rid_20, model_pid, model_color, linestyle='dashed')
            ax.plot(model_rid_80, model_pid, model_color, linestyle='dashed')

    def plot_slice(self, ax, pid_min, pid_max, censoring_limit=None):
        mask = (self.raw_pid > pid_min) & (self.raw_pid < pid_max)
        vals = self.raw_rid[mask]
        midpoint = np.mean((pid_min, pid_max))
        sns.ecdfplot(vals, color='C0', ax=ax)
        if censoring_limit:
            ax.axvline(x=censoring_limit, color='black')
            x = np.linspace(0.00, 0.05, 1000)
            y = self.evaluate_cdf(x, np.array((midpoint,)))
            ax.plot(x, y, color='C1')


class Model_P58(Model):
    """
    FEMA P-58 model.
    """

    def delta_fnc(self, pid: npt.NDArray, delta_y: float) -> npt.NDArray:
        delta = np.zeros_like(pid)
        mask = (delta_y <= pid) & (pid < 4.0 * delta_y)
        delta[mask] = 0.30 * (pid[mask] - delta_y)
        mask = 4.0 * delta_y <= pid
        delta[mask] = pid[mask] - 3.00 * delta_y
        return delta

    def set(self, *args, delta_y=0.00, beta=0.00, **kwargs) -> None:
        """
        The P-58 model requires the user to specify the parameters
        directly.
        """
        self.parameters = np.array((delta_y, beta))

    def evaluate_inverse_cdf(self, quantile: float, pid: npt.NDArray) -> npt.NDArray:
        """
        Evaluate the inverse of the conditional RID|PID CDF.
        """
        assert self.parameters is not None
        delta_y, beta = self.parameters
        delta_val = self.delta_fnc(pid, delta_y)
        return delta_val * np.exp(-np.sqrt(2.00) * beta * erfcinv(2.00 * quantile))

    def evaluate_cdf(self, rid: npt.NDArray, pid: npt.NDArray) -> npt.NDArray:
        """
        Evaluate the conditional RID|PID CDF.
        """
        assert self.parameters is not None
        delta_y, beta = self.parameters
        delta_val = self.delta_fnc(pid, delta_y)
        return 0.50 * erfc(-((np.log(rid / delta_val))) / (np.sqrt(2.0) * beta))


class Model_SP3(Model):
    """
    HB Risk SP3 model.
    """

    def delta_fnc(self, pid: npt.NDArray) -> npt.NDArray:
        delta = np.zeros_like(pid)
        assert self.parameters is not None
        c_1, c_2, c_3, c_4, delta_y, _ = self.parameters
        mask = (delta_y <= pid) & (pid < c_4 * delta_y)
        delta[mask] = c_1 * (pid[mask] - delta_y)
        mask = c_4 * delta_y <= pid
        delta[mask] = c_2 * pid[mask] - c_3 * delta_y
        return delta

    def set(
        self, *args, model_option: str, delta_y: float, dispersion: float, **kwargs
    ) -> None:
        """
        Sets the model parameters from a set of predefined values.

        Parameters
        ----------
        model_option: str
            Any of {elastic_plastic, general_inelastic, brbf_no_backup,
            brbf_no_backup_grav, brbf_backup}. Controls the median
            RID|PID.
        dispersion: float
            Dispersion for RID|PID.

        """

        parameters = {
            'elastic_plastic': np.array((0.25, 0.8, 3.0, 5.0)),
            'general_inelastic': np.array((0.125, 0.5, 2.0, 5.0)),
            'brbf_no_backup': np.array((0.25, 0.8, 3.0, 5.0)),
            'brbf_no_backup_grav': np.array((0.2, 0.5, 2.0, 6.0)),
            'brbf_backup': np.array((0.06, 0.425, 2.25, 6.0)),
        }

        if model_option not in parameters:
            raise ValueError(f'Invalid option: `{model_option}`')

        self.parameters = np.append(
            parameters[model_option], np.array((delta_y, dispersion))
        )

    def evaluate_inverse_cdf(self, quantile: float, pid: npt.NDArray) -> npt.NDArray:
        """
        Evaluate the inverse of the conditional RID|PID CDF.
        """
        assert self.parameters is not None
        delta_val = self.delta_fnc(pid)
        beta = self.parameters[-1]
        return delta_val * np.exp(-np.sqrt(2.00) * beta * erfcinv(2.00 * quantile))

    def evaluate_cdf(self, rid: npt.NDArray, pid: npt.NDArray) -> npt.NDArray:
        """
        Evaluate the conditional RID|PID CDF.
        """
        assert self.parameters is not None
        delta_val = self.delta_fnc(pid)
        beta = self.parameters[-1]
        return 0.50 * erfc(-((np.log(rid / delta_val))) / (np.sqrt(2.0) * beta))


class BilinearModel(Model):
    """
    One parameter constant, the other varies in a bilinear fashion.
    """

    def bilinear_fnc(self, pid: npt.NDArray) -> npt.NDArray:
        assert self.parameters is not None
        theta_1_a, c_lambda_slope, _ = self.parameters
        c_lambda_0 = 1.0e-8
        lambda_values = np.ones_like(pid) * c_lambda_0
        mask = pid >= theta_1_a
        lambda_values[mask] = (pid[mask] - theta_1_a) * c_lambda_slope + c_lambda_0
        return lambda_values

    def evaluate_inverse_cdf(self, quantile: float, pid: npt.NDArray):
        """
        Evaluate the inverse of the conditional RID|PID CDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_pdf(self, rid, pid, censoring_limit=None):
        """
        Evaluate the conditional RID|PID PDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_cdf(self, rid, pid):
        """
        Evaluate the conditional RID|PID CDF.
        """
        raise NotImplementedError("Subclasses should implement this.")


class TrilinearModel(Model):
    """
    One parameter constant, the other varies in a trilinear fashion.
    """

    def trilinear_fnc(self, pid: npt.NDArray, y0, m0, m1, m2, x0, x1) -> npt.NDArray:
        y1 = y0 + m0 * x0
        y2 = y1 + m1 * (x1 - x0)
        res = m0 * pid + y0
        mask = pid > x0
        res[mask] = (pid[mask] - x0) * m1 + y1
        mask = pid > x1
        res[mask] = (pid[mask] - x1) * m2 + y2
        return res

    def evaluate_inverse_cdf(self, quantile: float, pid: npt.NDArray):
        """
        Evaluate the inverse of the conditional RID|PID CDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_pdf(self, rid, pid, censoring_limit=None):
        """
        Evaluate the conditional RID|PID PDF.
        """
        raise NotImplementedError("Subclasses should implement this.")

    def evaluate_cdf(self, rid, pid):
        """
        Evaluate the conditional RID|PID CDF.
        """
        raise NotImplementedError("Subclasses should implement this.")


class Model_1_Weibull(BilinearModel):
    """
    Weibull model
    """

    def __init__(self) -> None:
        super().__init__()
        # initial parameters
        self.parameters = np.array((0.008, 0.30, 1.30))
        # parameter names
        self.parameter_names = ['pid_0', 'lambda_slope', 'kappa']
        # bounds
        self.parameter_bounds = [(0.00, 0.02), (0.00, 1.00), (0.80, 4.00)]

    def evaluate_pdf(
        self,
        rid: npt.NDArray,
        pid: npt.NDArray,
        censoring_limit: Optional[float] = None,
    ) -> npt.NDArray:
        assert self.parameters is not None
        _, _, theta_3 = self.parameters
        bilinear_fnc_val = self.bilinear_fnc(pid)
        pdf_val = sp.stats.weibull_min.pdf(rid, theta_3, 0.00, bilinear_fnc_val)
        pdf_val[pdf_val < 1e-6] = 1e-6
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(
                np.full(len(pid), censoring_limit), pid
            )
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        return pdf_val

    def evaluate_cdf(self, rid: npt.NDArray, pid: npt.NDArray) -> npt.NDArray:
        assert self.parameters is not None
        _, _, theta_3 = self.parameters
        bilinear_fnc_val = self.bilinear_fnc(pid)
        return sp.stats.weibull_min.cdf(rid, theta_3, 0.00, bilinear_fnc_val)

    def evaluate_inverse_cdf(self, quantile: float, pid: npt.NDArray) -> npt.NDArray:
        assert self.parameters is not None
        _, _, theta_3 = self.parameters
        bilinear_fnc_val = self.bilinear_fnc(pid)
        return sp.stats.weibull_min.ppf(quantile, theta_3, 0.00, bilinear_fnc_val)


class Model_Weibull_Trilinear(TrilinearModel):
    """
    Weibull model
    """

    def __init__(self) -> None:
        super().__init__()
        # initial parameters
        self.parameters = np.array((0.005, 0.15, 0.020, 0.40, 1.20, 1.20))
        # parameter names
        self.parameter_names = [
            'pid_0',
            'lambda_slope_0',
            'pid_1',
            'lambda_slope_1',
            'kappa_0',
            'kappa_1',
        ]
        # bounds
        self.parameter_bounds = [
            (0.00, 1.0e3),
            (0.00, 0.20),
            (0.00, 1.0e3),
            (0.00, 1.0e3),
            (0.01, 1.0e3),
            (0.01, 1.0e3),
        ]

    def obtain_lambda_and_kappa(
        self,
        pid: npt.NDArray,
    ) -> npt.NDArray:
        assert self.parameters is not None
        (
            pid_0,  # x0
            lambda_slope_0,  # m1
            pid_1,  # x1
            lambda_slope_1,  # m2
            kappa_0,  # y0, y1
            kappa_1,  # y2
        ) = self.parameters
        # calculate m0 for kappa
        kappa_slope_0 = (kappa_1 - kappa_0) / (pid_1 - pid_0)
        lambda_trilinear = self.trilinear_fnc(
            pid, 1e-6, 0.00, lambda_slope_0, lambda_slope_1, pid_0, pid_1
        )
        kappa_trilinear = self.trilinear_fnc(
            pid, kappa_0, 0.00, kappa_slope_0, 0.00, pid_0, pid_1
        )
        return lambda_trilinear, kappa_trilinear

    def evaluate_pdf(
        self,
        rid: npt.NDArray,
        pid: npt.NDArray,
        censoring_limit: Optional[float] = None,
    ) -> npt.NDArray:
        lambda_trilinear, kappa_trilinear = self.obtain_lambda_and_kappa(pid)
        pdf_val = sp.stats.weibull_min.pdf(
            rid, kappa_trilinear, 0.00, lambda_trilinear
        )
        pdf_val[pdf_val < 1e-6] = 1e-6
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(
                np.full(len(pid), censoring_limit), pid
            )
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        return pdf_val

    def evaluate_cdf(self, rid: npt.NDArray, pid: npt.NDArray) -> npt.NDArray:
        lambda_trilinear, kappa_trilinear = self.obtain_lambda_and_kappa(pid)
        return sp.stats.weibull_min.cdf(rid, kappa_trilinear, 0.00, lambda_trilinear)

    def evaluate_inverse_cdf(self, quantile: float, pid: npt.NDArray) -> npt.NDArray:
        lambda_trilinear, kappa_trilinear = self.obtain_lambda_and_kappa(pid)
        return sp.stats.weibull_min.ppf(
            quantile, kappa_trilinear, 0.00, lambda_trilinear
        )


class Model_2_Gamma(BilinearModel):
    """
    Gamma model
    """

    def __init__(self) -> None:
        super().__init__()
        # initial parameters
        self.parameters = np.array((0.008, 0.30, 1.30))
        # parameter names
        self.parameter_names = ['pid_0', 'lambda_slope', 'kappa']
        # bounds
        self.parameter_bounds = [(0.00, 0.02), (0.00, 1.00), (0.80, 4.00)]

    def evaluate_pdf(
        self,
        rid: npt.NDArray,
        pid: npt.NDArray,
        censoring_limit: Optional[float] = None,
    ) -> npt.NDArray:
        assert self.parameters is not None
        _, _, theta_3 = self.parameters
        bilinear_fnc_val = self.bilinear_fnc(pid)
        pdf_val = sp.stats.gamma.pdf(rid, theta_3, 0.00, bilinear_fnc_val)
        pdf_val[pdf_val < 1e-6] = 1e-6
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(
                np.full(len(pid), censoring_limit), pid
            )
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        return pdf_val

    def evaluate_cdf(self, rid: npt.NDArray, pid: npt.NDArray) -> npt.NDArray:
        assert self.parameters is not None
        _, _, theta_3 = self.parameters
        bilinear_fnc_val = self.bilinear_fnc(pid)
        return sp.stats.gamma.cdf(rid, theta_3, 0.00, bilinear_fnc_val)

    def evaluate_inverse_cdf(self, quantile: float, pid: npt.NDArray) -> npt.NDArray:
        assert self.parameters is not None
        _, _, theta_3 = self.parameters
        bilinear_fnc_val = self.bilinear_fnc(pid)
        return sp.stats.gamma.ppf(quantile, theta_3, 0.00, bilinear_fnc_val)


class Model_3_Beta(BilinearModel):
    """
    Beta model
    """

    def __init__(self) -> None:
        super().__init__()
        # initial parameters
        self.parameters = np.array((0.004, 100.00, 500.00))
        # parameter names
        self.parameter_names = ['pid_0', 'alpha_slope', 'beta']
        # bounds
        self.parameter_bounds = None

    def evaluate_pdf(
        self,
        rid: npt.NDArray,
        pid: npt.NDArray,
        censoring_limit: Optional[float] = None,
    ) -> npt.NDArray:
        assert self.parameters is not None
        _, _, c_beta = self.parameters
        c_alpha = self.bilinear_fnc(pid)
        pdf_val = sp.stats.beta.pdf(rid, c_alpha, c_beta)
        pdf_val[pdf_val < 1e-6] = 1e-6
        if censoring_limit:
            censored_range_mass = self.evaluate_cdf(
                np.full(len(pid), censoring_limit), pid
            )
            mask = rid <= censoring_limit
            pdf_val[mask] = censored_range_mass[mask]
        return pdf_val

    def evaluate_cdf(self, rid: npt.NDArray, pid: npt.NDArray) -> npt.NDArray:
        assert self.parameters is not None
        _, _, c_beta = self.parameters
        c_alpha = self.bilinear_fnc(pid)
        return sp.stats.beta.cdf(rid, c_alpha, c_beta)

    def evaluate_inverse_cdf(self, quantile: float, pid: npt.NDArray) -> npt.NDArray:
        assert self.parameters is not None
        _, _, c_beta = self.parameters
        c_alpha = self.bilinear_fnc(pid)
        return sp.stats.beta.ppf(quantile, c_alpha, c_beta)
