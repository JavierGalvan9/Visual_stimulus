#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:51:02 2021

@author: jgalvan
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from tempfile import TemporaryDirectory
from pathlib import Path
import argparse

parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parentDir, "Utils"))
from file_management import save_lzma

class FullFieldFlash:
    """
    The drifting grating stimulus consists of a full-field sinusoidal grating
    that drifts in a direction perpendicular to the orientation of the grating.
    This class generates a drifting gratings movie between two periods of
    gray/halt gratings frame with chosen orientation, spatial frequency and temporal 
    frequency. The timestep is by default set to 1 ms. Also, it converts the movie to a 
    train of instantaneous firing rates using the LGN module.

    Attributes:
    ----------
    - black_to_white: if True there would be a white flash after a black screen.
    Else there would a black flash after a white screen.
    - init_screen_dur: time interval in seconds of the first frame. 
    The default is 0.5
    - flash_dur: time interval in seconds of the flash.
    The default is 2.
    - end_screen_dur: time interval in seconds of the end frame. 
    The default is 0
    - min_value: minimum numerical value of the gratings. The default is -1 (black)
    - max_value: maximum numerical value of the gratings. The default is 1 (white).
    - contrast: ratio between the actual maximum and minimum value of the stimulus
    and the range max_value - min_value. The default is 0.8
    - dt: temporal resolution in seconds. The default is 0.001.
    - col_size: width dimension of the frame in pixels. The default is 240.
    - row_size: height dimension of the frame in pixels. The default is 120.
    """

    def __init__(self, black_to_white=True, init_screen_dur=0.5, flash_dur=1, end_screen_dur=0.5, 
                 min_value=-1, max_value=1, contrast=0.8, dt=0.001, col_size=240, row_size=120):
        self._black_to_white = black_to_white
        self._init_screen_dur = init_screen_dur
        self._flash_dur = flash_dur
        self._end_screen_dur = end_screen_dur
        self._min_value = min_value
        self._max_value = max_value
        self._contrast = contrast
        self._dt = dt
        self._height = row_size
        self._width = col_size
        # time until the beginning of the visual flow
        self.first_transition = int(self._init_screen_dur/self._dt)
        # time until the end of the visual flow
        self.second_transition = int((self._init_screen_dur+self._flash_dur)/self._dt)
        # total movie time
        self.total_time = int(
            (self._init_screen_dur + self._flash_dur + self._end_screen_dur)/self._dt)
        self.flash_time = int(self._flash_dur/self._dt)
        self.init_screen_time = int(self._init_screen_dur/self._dt)
        self.end_screen_time = int(self._end_screen_dur/self._dt)
        self.visual_stimulus = np.zeros(
            (self.total_time, self._height, self._width))
        
        # Create a filename according to the attributes of the class
        filename_items = []
        for attr, value in self.__dict__.items():
            if attr.startswith('_'):
                filename_items.append(str(attr[1:])+'_'+str(value))
        self.filename = '&'.join(filename_items)
        
    def black_frame(self):
        black_frame = np.zeros((self._height, self._width))  
        black_frame = self._min_value*self._contrast
        return black_frame
    
    def white_frame(self):
        white_frame = np.zeros((self._height, self._width))
        white_frame = self._max_value*self._contrast
        return white_frame

    def visual_stimulus_generation(self):
        """
        This function generates the full field flash stimulus, either black to
        white or white to black.
        """
        
        white_frame_array = self.white_frame()
        black_frame_array = self.black_frame()
        
        if self._black_to_white:
            self.visual_stimulus[:self.first_transition, :, :] = black_frame_array
            self.visual_stimulus[self.first_transition:self.second_transition, :, :] = white_frame_array
            self.visual_stimulus[self.second_transition:, :, :] = black_frame_array
        else:
            self.visual_stimulus[:self.first_transition, :, :] = white_frame_array
            self.visual_stimulus[self.first_transition:self.second_transition, :, :] = black_frame_array
            self.visual_stimulus[self.second_transition:, :, :] = white_frame_array

    def create_visual_stimulus(self, save_init_frame=False, save_stimulus_array=False, save_movie=False, create_lgn_firing_rates=False):
        """
        This function creates the visual stimulus.

        Parameters
        ----------
        save_init_frame : TYPE, optional
            DESCRIPTION. Allows the user to save a .png image of the initial
            frame given by the initial_frame method.
            The default is False.
        save_stimulus_array : TYPE, optional
            DESCRIPTION. Allows the user to save a .lzma file with the array
            of the stimulus values.
            The default is False.
        save_movie : TYPE, optional
            DESCRIPTION. Allows the user to save a movie of the visual stimulus.
            The default is False.
        create_lgn_firing_rates : TYPE, optional
            DESCRIPTION. Allows the user to create firing rates using the LGN
            module from the Billeh model. This requires the stimulus size to be
            (240, 120), the dt=1 and the min_value and max_value within -1 and
            1.
            The default is False.

        Returns
        -------
        None.

        """
        
        self.visual_stimulus_generation()
            
        if save_init_frame:
            fig = plt.figure()
            fig.set_size_inches(1. * self._width / self._height, 1, forward = False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.visual_stimulus[0, :, :], 
                      vmin=self._min_value, vmax= self._max_value,
                      interpolation='nearest', cmap="binary_r")
            Path('Initial_frame').mkdir(parents=True, exist_ok=True)
            fig.savefig(os.path.join('Initial_frame', self.filename+'.png'),
                       dpi = self._width)
            plt.close()
        
        if save_stimulus_array:
            Path('Stimulus_array').mkdir(parents=True, exist_ok=True)
            filename = self.filename+'.lzma'
            save_lzma(self.visual_stimulus, filename, 'Stimulus_array')

        if save_movie:
            Path('Video').mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory() as Drifting_gratings_images:
                video_name = os.path.join('Video', self.filename+'.avi')
                files = []
                fig, ax = plt.subplots(figsize=(self._height, self._width))
                movie_data = (self.visual_stimulus - self._min_value)/(self._max_value - self._min_value)
                for i in range(self.total_time):
                    img = Image.fromarray((movie_data[i, :, :]*255).astype(np.uint8), 'L')
                    fname = '_tmp%03d.png' % i
                    img.save(os.path.join(Drifting_gratings_images, fname))
                    files.append(fname)
                video = cv2.VideoWriter(video_name, 0, int(
                    1/self._dt), (self._width, self._height))
                for image in files:
                    video.write(cv2.imread(os.path.join(
                        Drifting_gratings_images, image)))
                cv2.destroyAllWindows()
                video.release()

        if create_lgn_firing_rates:
            
            import lgn 
            
            visual_stimulus = self.visual_stimulus[..., None].astype(np.float32)
            lgn_model = lgn.LGN()
            t0 = time.time()
            spatial = lgn_model.spatial_response(visual_stimulus)  # (total_time, 17400)
            firing_rates = lgn_model.firing_rates_from_spatial(*spatial)
            np_firing_rates = firing_rates.numpy()
            print(np_firing_rates.shape)
            print(f'Computed LGN response to visual stimuli in {time.time() - t0:.2f} seconds')
            print(f'Converted stimulus {visual_stimulus.shape} (video) to LGN firing rates {np_firing_rates.shape}')
            Path('LGN_firing_rates').mkdir(parents=True, exist_ok=True)
            save_lzma(np_firing_rates, self.filename+'.lzma', 'LGN_firing_rates')
        
        
def main(flags):
    
    instantiate_stimulus = FullFieldFlash(black_to_white=flags.black_to_white, init_screen_dur=flags.init_screen_dur,
                                          flash_dur=flags.flash_dur, end_screen_dur=flags.end_screen_dur,
                                          contrast=flags.contrast)
    instantiate_stimulus.create_visual_stimulus(save_init_frame=flags.save_init_frame, save_stimulus_array=flags.save_stimulus_array, 
                                                save_movie=flags.save_movie, create_lgn_firing_rates=flags.create_lgn_firing_rates)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define some flags')
    
    parser.add_argument('--init_screen_dur', type=float, default=0.5)
    parser.add_argument('--flash_dur', type=float, default=2)
    parser.add_argument('--end_screen_dur', type=float, default=0)
    parser.add_argument('--contrast', type=float, default=0.8)
    parser.add_argument('--black_to_white', action='store_true')
    parser.add_argument('--no-black_to_white', dest='black_to_white', action='store_false')
    parser.set_defaults(black_to_white=True)
    parser.add_argument('--save_init_frame', action='store_true')
    parser.add_argument('--no-save_init_frame', dest='save_init_frame', action='store_false')
    parser.set_defaults(save_init_frame=False)
    parser.add_argument('--save_stimulus_array', action='store_true')
    parser.add_argument('--no-save_stimulus_array', dest='save_stimulus_array', action='store_false')
    parser.set_defaults(save_stimulus_array=False)
    parser.add_argument('--save_movie', action='store_true')
    parser.add_argument('--no-save_movie', dest='save_movie', action='store_false')
    parser.set_defaults(save_movie=True)
    parser.add_argument('--create_lgn_firing_rates', action='store_true')
    parser.add_argument('--no-create_lgn_firing_rates', dest='create_lgn_firing_rates', action='store_false')
    parser.set_defaults(create_lgn_firing_rates=True)
    
    flags = parser.parse_args()
    
    main(flags)
    
