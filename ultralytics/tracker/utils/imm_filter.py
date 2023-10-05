import numpy as np
from .imm_utils.filters.iEKF import iEKF
from .imm_utils.dynamics_models.CA import CA
from .imm_utils.dynamics_models.CV import CV
from .imm_utils.dynamics_models.CT import CT
from .imm_utils.measurement_models.range_bearing import RangeBearing
from .imm_utils.measurement_models.range_only import RangeOnly
from .imm_utils.imm.Gaussian_Mixture import GaussianMixture
from .imm_utils.imm.IMM import IMM
from .imm_utils.utils.Gauss import GaussState
import scipy.linalg
sigma_dict=dict(
sigma_q = 0.05,
sigma_r = 0.1,
sigma_th = 0.0005,
sigma_a = 0.03,
sigma_w = 0.02
)


class IMMFilter(object):

    def __init__(self) -> None:
        
        self.sigma_q = sigma_dict["sigma_q"]
        self.sigma_r = sigma_dict["sigma_r"]
        self.sigma_th = sigma_dict["sigma_th"]
        self.sigma_a = sigma_dict["sigma_a"]
        self.sigma_w = sigma_dict["sigma_w"]
        self.sensor_model = RangeBearing(self.sigma_r, self.sigma_th, state_dim=7)

        self.filters = [
            iEKF(CV(self.sigma_q*10), self.sensor_model),
            iEKF(CT(self.sigma_a*1, sigma_w=self.sigma_w), self.sensor_model),
            iEKF(CA(sigma=self.sigma_a*20), self.sensor_model),]
        
        self.init_means=[]
        self.init_covs=[]
        self.n_filt=3
        self.alpha=0.8
        self.PI = self.generate_pi()
        self.imm=IMM(self.filters,self.PI)

    def initiate(self, measurement):
        mean_pos=measurement[:2]
        mean_pos=np.array(mean_pos.tolist()+[0,0,0,0,0])
        self.init_means=[mean_pos,mean_pos,mean_pos]

        for i in range(self.n_filt):
            init_cov1 = np.eye((7))*0.1
            init_cov1[6, 6] = 1e-2
            self.init_covs.append(init_cov1)

        self.init_weights = np.ones((self.n_filt, 1))/self.n_filt

        self.init_states = [GaussState(init_mean, init_cov) for init_mean, init_cov in zip(self.init_means, self.init_covs)]

        self.immstate = GaussianMixture(
            self.init_weights, self.init_states
        )
        self.PI = self.generate_pi()
        self.imm=IMM(self.filters,self.PI)
        gauss0,_=self.imm.get_estimate(self.immstate)
        gauss0.mean[2]=measurement[2]
        gauss0.mean[3]=measurement[3]
        return gauss0.mean.reshape(1,-1).squeeze(),gauss0.cov,self.immstate



    def generate_pi(self):
        Pi = np.eye(self.n_filt)*self.alpha
        for i in range(self.n_filt):
            for j in range(self.n_filt):
                if i != j:
                    Pi[i, j] = (1-self.alpha)/(self.n_filt-1)
        return Pi



    def predict(self,mean,convariance,immstate):
        self.immstate=self.imm.predict(immstate,u=None,T=1)
        gauss,_=self.imm.get_estimate(self.immstate)
        # gauss.mean[0]=mean[0]
        # gauss.mean[1]=mean[1]
        gauss.mean[2]=mean[2]
        gauss.mean[3]=mean[3]
        return gauss.mean.reshape(1,-1).squeeze(),gauss.cov,self.immstate




    def multi_predict(self, means, covariances,immstates):
        gauss_list=[]
        cov_list=[]
        imm_list=[]
        for imm_each,mean,cov in zip(immstates,means,covariances):

            immstate=self.imm.predict(imm_each,u=None,T=1)
            gauss,_=self.imm.get_estimate(immstate)
            # gauss.mean[0]=mean[0]
            # gauss.mean[1]=mean[1]
            gauss.mean[2]=mean[2]
            gauss.mean[3]=mean[3]
            gauss_list.append(gauss.mean.reshape(1,-1).squeeze())
            cov_list.append(gauss.cov)
            imm_list.append(immstate)
        return gauss_list,cov_list,imm_list

    def update(self, mean, covariance, measurement,immstate):
        measurement_two=measurement[:2]
        mean_pos=np.array(measurement_two.tolist()+[0,0,0,0,0])

        measurement_two=self.sensor_model.h(mean_pos)

        # measurement_two=measurement_two[:,None]
        self.immstate=self.imm.update(immstate,measurement_two)
        
        gauss,_=self.imm.get_estimate(self.immstate)
        # print(gauss.mean)
        # gauss.mean[0]=measurement[0]
        # gauss.mean[1]=measurement[1]
        gauss.mean[2]=measurement[2]
        gauss.mean[3]=measurement[3]

        return gauss.mean.reshape(1,-1).squeeze(),gauss.cov,self.immstate
    
    def gating_distance(self, mean, covariance, measurements,
                        only_position=True):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        d = measurements[:,:2] - mean[:2]
        squared_maha = np.sqrt(np.sum(d * d, axis=1))/20
        # print(measurements,mean,squared_maha)
        return squared_maha
    