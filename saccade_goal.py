import numpy as np
from load_data import load_data


def random_dots(N=32):
    """
    Computes random "saliency map" with shape (N,N)
    """
    return np.random.random((N,N))



class Saccade_goal():

    def __init__(self, _WIDTH=32, _fov=24):
        """
        _WIDTH: Width/Height of saliency map
        _fov: Field of view

        Here we compute a mask to apply on an incomming saliency map. The mask only has to be computed once. 
        """
        # Save width & fov
        self.WIDTH = _WIDTH
        self.fov = _fov

        # List of indices of future saliency map
        self.index_list = np.arange(_WIDTH**2)

        # Compute midpoint for odd/even width
        if self.WIDTH % 2 != 0:
            self.MIDPOINT = self.WIDTH // 2
        else:
            self.MIDPOINT = self.WIDTH // 2 + .5 - 1

        # Map with ccentricities from midpoint in pixel
        self.eccentricity_map = np.fromfunction(lambda i, j: np.sqrt((i-self.MIDPOINT)**2 + (j-self.MIDPOINT)**2), (_WIDTH, _WIDTH))
        # convert into degrees
        self.eccentricity_map *= _fov/(_WIDTH-1)

        # Count and save how often eccentricity in certain eccentricity bin accures as map       
        frequency, bins = np.histogram(self.eccentricity_map.flatten(), range=(0,_fov*np.sqrt(2)+1), bins=30)
        self.eccentricity_frequency_map = np.zeros((_WIDTH, _WIDTH))
        for i in range(_WIDTH):
            for j in range(_WIDTH):
                ecc = self.eccentricity_map[i,j]
                bin_index = np.digitize(ecc, bins)-1 # -1 to convert to array index & -1 because #Bins = #Datapoints+1
                self.eccentricity_frequency_map[i,j] =  frequency[bin_index]


        # Load Infant data from Bambach et al. analysis (mean_ch)
        # Hist range from 0 to biggest possible amplitude+1°, n_bins picked by hand such histogram is smooth line
        xdata, (mean_ch, error_ch), (mean_pa, error_pa), (mean_schütt, error_schütt) = load_data(hist_range=(0,_fov*np.sqrt(2)+1), n_bins=30)


        # Compute mask with probability value for each eccentricity 
        self.data_mask = np.zeros((_WIDTH, _WIDTH))
        for i in range(_WIDTH):
            for j in range(_WIDTH):
                ecc = self.eccentricity_map[i,j]
                index = np.digitize(ecc, xdata)
                self.data_mask[i,j] = mean_ch[index]

        # Compute overall mask, which gets applied to future saliency map
        # Mask uniforms the prob. to pick a pixel at certain eccentricity bin and multiplies this with the prob. density from the data
        self.data_mask = self.data_mask / self.eccentricity_frequency_map
        self.magnification_mask = self.eccentricity_map * (1/(0.77+self.eccentricity_map))**2 #dAde = e M**2


    def saccade_goal(self, sal_map, mask_type='magnification', output_mode='angles'):
        """
        sal_map: Saliency map with shape (self.WIDTH, self.WIDTH)
        mode: Output mode
                - 'index'->Tuple: Array index of saccade goal in frame of the saliency map
                - 'eccentricity'->Float: Eccentricity of saccade goal
                - 'angles'->Tuple: Horizontal & vertical angle of saccade goal in degrees
        """
        
        if mask_type=='data':
            mask = self.data_mask
        elif mask_type=='magnification':
            mask = self.magnification_mask

        # Apply mask
        sal_map *= mask
        # Normalize
        sal_map /= np.sum(sal_map)

        choice = np.random.choice(self.index_list, p=sal_map.flatten())
        # saccade goal as array index
        goal_index = np.divmod(choice, self.WIDTH)

        # print(choice)
        # print(sal_map.flatten()[choice], sal_map[goal_index])) # If correct this should output the same number twice
        # print('Goal index', goal_index)
        # print('Eccentricity', self.eccentricity_map[goal_index])
        # print('Angles',((goal_index[0]-self.MIDPOINT)*self.fov/(self.WIDTH-1), -(goal_index[1]-self.MIDPOINT)*self.fov/(self.WIDTH-1)))

        if output_mode=='index':
            return goal_index
        elif output_mode=='eccentricity':
            return self.eccentricity_map[goal_index]
        elif output_mode=='angles':
            return ((goal_index[0]-self.MIDPOINT)*self.fov/(self.WIDTH-1), -(goal_index[1]-self.MIDPOINT)*self.fov/(self.WIDTH-1)) # Vertical axis is inverted in numpy arrays! -> 2'nd component x(-1)

    