# Authors:
# - Nathanyel Schut (12907995)
# - Frenk Klein Schiphorst (11866497)

"""Module for creating images and area estimations for the Mandelbrot set"""
import math
from dataclasses import dataclass

import matplotlib
import numpy as np
from PIL import Image
from scipy.stats import qmc

def pure_random_sample(
        s: int, re_bounds: tuple, im_bounds: tuple) -> np.ndarray:
    """Generate a uniform (pseudo) random complex sample between the 
    given bounds.

    Args:
        s (int): number of sample points
        re_bounds (tuple): bounds for the real part of the complex 
                            numbers
        im_bounds (tuple): bounds for the imaginary part of the complex
                            numbers

    Returns:
        np.ndarray: 1D numpy array of size s, containing randomly 
                    generated complex numbers
    """
    return (np.random.uniform(re_bounds[0], re_bounds[1], size=s)
            + np.random.uniform(im_bounds[0], im_bounds[1], size=s) * 1j)

def latin_hypercube_sample(
        s: int, re_bounds: tuple, im_bounds: tuple) -> np.ndarray:
    """Generate a quasi-random Latin hypercube complex sample between
    the given bounds.

    Args:
        s (int): number of sample points
        re_bounds (tuple): bounds for the real part of the complex 
                            numbers
        im_bounds (tuple): bounds for the imaginary part of the complex
                            numbers

    Returns:
        np.ndarray: 1D numpy array of size s, containing quasi-random 
                    complex numbers
    """
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=s)

    sample = qmc.scale(sample, [re_bounds[0], im_bounds[0]], [re_bounds[1], im_bounds[1]])

    return sample[:, 0] + sample[:, 1] * 1j

def orthogonal_sample(
    s: int, re_bounds: tuple, im_bounds: tuple) -> np.ndarray:
    """Generate a quasi-random orthogonal complex sample between the
    given bounds.

    Args:
        s (int): number of sample points
        re_bounds (tuple): bounds for the real part of the complex 
                            numbers
        im_bounds (tuple): bounds for the imaginary part of the complex
                            numbers

    Returns:
        np.ndarray: 1D numpy array of size s, containing quasi-random 
                    complex numbers
    """    
    sampler = qmc.LatinHypercube(d=2, strength=2)
    sample = sampler.random(n=s)

    sample = qmc.scale(sample, [re_bounds[0], im_bounds[0]], [re_bounds[1], im_bounds[1]])

    return sample[:, 0] + sample[:, 1] * 1j

def test_sample(
    s: int, re_bounds: tuple, im_bounds: tuple) -> tuple:
    """_summary_

    Args:
        s (int): number of sample points
        re_bounds (tuple): bounds for the real part of the complex 
                            numbers
        im_bounds (tuple): bounds for the imaginary part of the complex
                            numbers

    Returns:
        tuple: size 2, containing 1D numpy arrays of size s, containing
                quasi-random complex numbers generated with anti-thetic
                variables
    """
    Orth_sample = orthogonal_sample(s, re_bounds, im_bounds)

    Orth_sample_re = np.real(Orth_sample)
    Orth_sample_im = np.imag(Orth_sample)

    U_re = (Orth_sample_re - re_bounds[0]) / abs(re_bounds[1] - re_bounds[0])
    U_im = (Orth_sample_im - im_bounds[0]) / abs(im_bounds[1] - im_bounds[0])

    U_re2 = 1 - U_re
    U_im2 = 1 - U_im

    Orth_sample_re2 = U_re2 * abs(re_bounds[1] - re_bounds[0]) + re_bounds[0]
    Orth_sample_im2 = U_im2 * abs(im_bounds[1] - im_bounds[0]) + im_bounds[0]

    Orth_sample2 = Orth_sample_re2 + 1j * Orth_sample_im2

    return Orth_sample, Orth_sample2

@dataclass
class MandelbrotSet:
    """Dataclass for working with the Mandelbrot set. Allows for image
    creation and area estimation.

    Args:
        i (int): number of iterations
        s (int): number of sample points
        re_bounds (tuple): bounds for the real part of the complex 
                            numbers
        im_bounds (tuple): bounds for the imaginary part of the complex
                            numbers
        members (int, default=0): variable to keep track of the amount
                                    of numbers in the set
    """
    i: int
    s: int
    re_bounds: tuple
    im_bounds: tuple
    members: int = 0

    def __contains__(self, c: complex) -> bool:
        """Function for checking membership of the Mandelbrot set.

        Args:
            c (complex): number to check

        Returns:
            bool: indication of membership
        """        
        z = 0
        for _ in range(self.i):
            z = z**2 + c
            if abs(z) > 2:
                return False
    
        return True

    def _stability_(self, c: complex) -> float:
        """Float number indicating the normalized number of iterations
        to exclude c from the set.

        Args:
            c (complex): number for which to get stability

        Returns:
            float: stability
        """        
        z = 0
        for _i in range(self.i):
            z = z**2 + c
            if abs(z) > 2:
                return _i / self.i
        
        return 1.0

    def _continuous_stability_(self, c: complex) -> float:
        """Float number indicating the normalized continuous number of
        iterations to exclude c from the set.

        Args:
            c (complex): number for which to get stability

        Returns:
            float: continuous stability
        """        
        z = 0
        for _i in range(self.i):
            z = z**2 + c
            if abs(z) > 2:
                value = (_i + 1 - math.log(math.log(abs(z))) / math.log(2)) / self.i
                return max(0.0, (min(value, 1.0)))
        
        return 1.0
    
    def _stability_to_color_(self, stability: float, palette: list) -> tuple:
        """Conversion from stability value to color values.

        Args:
            stability (float): stability value. Should be between 0, 1
            palette (list): color palette to use

        Returns:
            tuple: color values
        """        
        index = int(min(stability * len(palette), len(palette) - 1))
        colors = palette[index % len(palette)]

        return colors

    def create_image(self, pixel_width: int, pixel_height: int, cmap = 'plasma', banding: bool = False):
        """Creates an image of the Mandelbrot set.

        Args:
            pixel_width (int): number of pixels horizontally
            pixel_height (int): number of pixels vertically
            cmap (str, optional): colormap. Defaults to 'plasma'.
            banding (bool, optional): whether to use regular or 
                                        continuous stability.
                                        Defaults to False.

        Returns:
            PIL.Image: Pillow image
        """        
        re = np.linspace(self.re_bounds[0], self.re_bounds[1], pixel_width)
        im = np.linspace(self.im_bounds[0], self.im_bounds[1], pixel_height)
        c = re[np.newaxis, :] + im[:, np.newaxis] * 1j

        if banding:
            vec_stability = np.vectorize(self._stability_)
        
        else:
            vec_stability = np.vectorize(self._continuous_stability_)

        stability_matrix = vec_stability(c)
        
        if isinstance(cmap, str):
            stability_colors = np.uint8(matplotlib.colormaps[cmap](stability_matrix, bytes=True))
        elif isinstance(cmap, matplotlib.colors.ListedColormap):
            stability_colors = np.uint8(cmap(stability_matrix, bytes=True))
            
        stability_colors = stability_colors[:, :, :4]

        im = Image.fromarray(stability_colors, mode="RGBA")

        return im

    def area_estimate(self, sampling_method: str) -> float:
        """Function for estimating Mandelbrot set area.

        Args:
            sampling_method (str): which type of samples to use. Can
            choose between the following:
                - pure_random
                - latin_hypercube
                - orthogonal
                - test

        Returns:
            float: area estimate
        """        
        if sampling_method == 'pure_random':
            sampling_func = pure_random_sample

        elif sampling_method == 'latin_hypercube':
            sampling_func = latin_hypercube_sample

        elif sampling_method == 'orthogonal':
            sampling_func = orthogonal_sample
        
        elif sampling_method == 'test':
            c1, c2 = test_sample(self.s, self.re_bounds, self.im_bounds)
            c1_members = 0
            c2_members = 0

            for c1i in c1:
                if c1i in self:
                    c1_members += 1

            for c2i in c2:
                if c2i in self:
                    c2_members += 1

            total_area = (self.re_bounds[1] - self.re_bounds[0]) * (self.im_bounds[1] - self.im_bounds[0])
            area1 = total_area * (c1_members / self.s)
            area2 = total_area * (c2_members / self.s)

            return (area1 + area2) / 2

        c = sampling_func(self.s, self.re_bounds, self.im_bounds)
        for c_i in c:
            if c_i in self:
                self.members += 1
            
        return (self.re_bounds[1] - self.re_bounds[0]) * (self.im_bounds[1] - self.im_bounds[0]) * (self.members / self.s)
