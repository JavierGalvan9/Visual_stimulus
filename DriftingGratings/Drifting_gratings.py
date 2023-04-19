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


class DriftingGratings:
    """
    The drifting grating stimulus consists of a full-field sinusoidal grating
    that drifts in a direction perpendicular to the orientation of the grating.
    This class generates a drifting gratings movie between two periods of
    gray/halt gratings frame with chosen orientation, spatial frequency and temporal 
    frequency. The timestep is by default set to 1 ms. Also, it converts the movie to a 
    train of instantaneous firing rates using the LGN module.

    Attributes:
    ----------
    - orientation: orientation of the gratings in degrees.
    Possible values: 0, 45, 90, 135, 180, 225, 270, 315.
    The default is 0.
    - TF: temporal frequency of the drifting in Hertz. The default is 2 Hz.
    - SF: spatial frequency of the gratings in cycles/degree. The default is 0.04 cpd.
    - reverse: allows the user to reverse the stimulus: first moving image and 
    halt it during the stimulus. The default is False.
    - init_screen_dur: time interval in seconds of the first frame. 
    The default is 0.5
    - visual_flow_dur: time interval in seconds of the drifting gratings.
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
    - init_gray_screen: if True the first screen is a gray image, else it is 
    a halt static gratings image. The default is True.
    - end_gray_screen: if True the end screen is a gray image, else it is 
    a halt static gratings image. The default is True.
    """

    def __init__(self, orientation, TF=2, SF=0.04, reverse=False, init_screen_dur=0.5,
                 visual_flow_dur=2, end_screen_dur=0, min_value=-1, max_value=1, contrast=0.8,
                 dt=0.001, col_size=240, row_size=120, init_gray_screen=True, end_gray_screen=True):
        self._orientation = orientation
        self._TF = TF
        self._SF = SF
        self._reverse = reverse
        self._init_screen_dur = init_screen_dur
        self._visual_flow_dur = visual_flow_dur
        self._end_screen_dur = end_screen_dur
        self._min_value = min_value
        self._max_value = max_value
        self._contrast = contrast
        self._dt = dt
        self._height = row_size
        self._width = col_size
        self._init_gray_screen = init_gray_screen
        self._end_gray_screen = end_gray_screen
        # time until the beginning of the visual flow
        self.first_transition = int(self._init_screen_dur/self._dt)
        # time until the end of the visual flow
        self.second_transition = int(
            (self._init_screen_dur+self._visual_flow_dur)/self._dt)
        # total movie time
        self.total_time = int(
            (self._init_screen_dur + self._visual_flow_dur + self._end_screen_dur)/self._dt)
        self.visual_flow_time = int(self._visual_flow_dur/self._dt)
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

    def from_angle_to_gratings(self, angles):
        """
        This function converts an array of angles into an array of gratings
        values according to the min_value, max_value and contrast chosen.

        Parameters
        ----------
        angles : TYPE, np.ndarray

        Returns
        -------
        gratings : TYPE, np.ndarray

        """
        gratings = (((np.sin(angles)+1)/2) * (self._max_value -
                    self._min_value) + self._min_value)*self._contrast
        return gratings

    def initial_frame(self, gray_screen=True):
        """
        This function creates the initial frame according to the parameters 
        given in the class instatiation.

        Parameters
        ----------
        gray_screen : TYPE, optional
            DESCRIPTION. Allows the user to choose whether the frame is a gray
            screen or a static gratings frame.
            The default is True.

        Returns
        -------
        None.

        """

        init_frame = np.zeros((self._height, self._width))

        if not gray_screen:
            if self._orientation in [0, 180]:
                self.n_gratings = self._SF * self._width
                init_frame[:, :] += np.linspace(0,
                                                2*np.pi*self.n_gratings, self._width)
            elif self._orientation in [90, 270]:
                self.n_gratings = self._SF * self._height
                init_frame = np.transpose(np.transpose(
                    init_frame[:, :])+np.linspace(0, 2*np.pi*self.n_gratings, self._height))
            elif self._orientation in [45, 135, 225, 315]:
                #                 small_diagonal = np.sqrt(2*(min([self._height, self._width]))**2)
                #             self._n_gratings = self._SF * self._small_diagonal
                n_horizontal_gratings = (
                    self._SF/np.cos(np.deg2rad(self._orientation))) * self._width
    #             n_vertical_gratings = (self._SF/np.sin(np.deg2rad(self._orientation))) * self._height
                init_frame[0, :] += np.linspace(0, 2*np.pi*n_horizontal_gratings,
                                                self._width)/2
                delta_y = 2*np.pi*self._SF / \
                    np.sin(np.deg2rad(self._orientation))/2
                for i in range(1, self._height):
                    init_frame[i, :] = init_frame[i-1, :] + delta_y
            else:
                print(
                    'The orientation selected is not within the available ones. Try another one!')

        return init_frame

    def gratings(self):
        """
        This function generates the drifting gratings stimulus starting from a
        gray/halt_gratings screen, then the drifting gratings period, and finally
        another gray/halt_gratings screen.
        """
        # First period: static frame
        init_frame = self.initial_frame(gray_screen=self._init_gray_screen)
        self.visual_stimulus[:self.first_transition, :,
                             :] = self.from_angle_to_gratings(init_frame)

        # Second period: drifting gratings
        angle_shift = np.zeros(
            [self.visual_flow_time, self._height, self._width])
        angle_shift[:, :, :] = self.initial_frame(gray_screen=False)
        for i in range(self.visual_flow_time):
            if self._orientation in [180, 270]:
                angle_shift[i, :, :] += i*self._dt*self._TF*2*np.pi
            else:
                angle_shift[i, :, :] -= i*self._dt*self._TF*2*np.pi
        current_angles = angle_shift[-1, :, :]
        self.visual_stimulus[self.first_transition:self.second_transition,
                             :, :] = self.from_angle_to_gratings(angle_shift)

        # Third period: static frame
        if self._end_gray_screen:
            self.visual_stimulus[self.second_transition:, :, :] = self.from_angle_to_gratings(
                self.initial_frame(gray_screen=self._end_gray_screen))
        else:
            self.visual_stimulus[self.second_transition:, :,
                                 :] = self.from_angle_to_gratings(current_angles)

    def reversed_gratings(self):
        """
        This function generates the drifting gratings stimulus starting from the
        drifting gratings and halting them after the init_period, then intercede 
        a gray/halt_gratings screen, and finally return to drifting gratings.
        """

        # First period: drifting gratings
        init_frame = self.initial_frame(gray_screen=False)
        first_angle_shift = np.zeros(
            [self.init_screen_time, self._height, self._width])
        first_angle_shift[:, :, :] = init_frame
        for i in range(self.init_screen_time):
            if self._orientation in [180, 270]:
                first_angle_shift[i, :, :] += i*self._dt*self._TF*2*np.pi
            else:
                first_angle_shift[i, :, :] -= i*self._dt*self._TF*2*np.pi
        current_angles = first_angle_shift[-1, :, :]
        self.visual_stimulus[:self.first_transition, :,
                             :] = self.from_angle_to_gratings(first_angle_shift)

        # Second period: static frame
        if self._init_gray_screen:
            self.visual_stimulus[self.first_transition:self.second_transition,
                                 :, :] = self.from_angle_to_gratings(self.initial_frame())
        else:
            self.visual_stimulus[self.first_transition:self.second_transition,
                                 :, :] = self.from_angle_to_gratings(current_angles)

        # Third period: drifting gratings
        second_angle_shift = np.zeros(
            [self.end_screen_time, self._height, self._width])
        second_angle_shift[:, :, :] = current_angles
        for i in range(self.end_screen_time):
            if self._orientation in [180, 270]:
                second_angle_shift[i, :, :] += i*self._dt*self._TF*2*np.pi
            else:
                second_angle_shift[i, :, :] -= i*self._dt*self._TF*2*np.pi
        self.visual_stimulus[self.second_transition:, :,
                             :] = self.from_angle_to_gratings(second_angle_shift)

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
        t_stimulus = time.time()
        if self._reverse:
            self.reversed_gratings()
        else:
            self.gratings()
        print('Stimulus created in {:.2f} seconds'.format(
            time.time()-t_stimulus))
        if save_init_frame:
            fig = plt.figure()
            fig.set_size_inches(1. * self._width /
                                self._height, 1, forward=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(self.visual_stimulus[0, :, :],
                      vmin=self._min_value, vmax=self._max_value,
                      interpolation='nearest', cmap="binary_r")
            Path('Initial_frame').mkdir(parents=True, exist_ok=True)
            fig.savefig(os.path.join('Initial_frame', self.filename+'.png'),
                        dpi=self._width)
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
                movie_data = (self.visual_stimulus - self._min_value) / \
                    (self._max_value - self._min_value)
                for i in range(self.total_time):
                    img = Image.fromarray(
                        (movie_data[i, :, :]*255).astype(np.uint8), 'L')
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

            visual_stimulus = self.visual_stimulus[..., None].astype(
                np.float32)
            lgn_model = lgn.LGN(row_size=self._height, col_size=self._width)
            t0 = time.time()
            spatial = lgn_model.spatial_response(
                visual_stimulus)  # (total_time, 17400)
            firing_rates = lgn_model.firing_rates_from_spatial(*spatial)
            np_firing_rates = firing_rates.numpy()
            print(
                f'Computed LGN response to visual stimuli in {time.time() - t0:.2f} seconds')
            print(
                f'Converted stimulus {visual_stimulus.shape} (video) to LGN firing rates {np_firing_rates.shape}')
            Path('LGN_firing_rates').mkdir(parents=True, exist_ok=True)
            save_lzma(np_firing_rates, self.filename +
                      '.lzma', 'LGN_firing_rates')


def main(flags):

    instantiate_gratings = DriftingGratings(flags.gratings_orientation, TF=flags.temporal_frequency,
                                            SF=flags.spatial_frequency, reverse=flags.reverse, init_screen_dur=flags.init_screen_dur,
                                            visual_flow_dur=flags.visual_flow_dur, end_screen_dur=flags.end_screen_dur,
                                            contrast=flags.contrast, init_gray_screen=flags.init_gray_screen,
                                            end_gray_screen=flags.end_gray_screen)
    instantiate_gratings.create_visual_stimulus(save_init_frame=flags.save_init_frame, save_stimulus_array=flags.save_stimulus_array,
                                                save_movie=flags.save_movie, create_lgn_firing_rates=flags.create_lgn_firing_rates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define some flags')

    parser.add_argument('--gratings_orientation', type=int,
                        choices=range(0, 360, 45), default=0)
    parser.add_argument('--temporal_frequency', type=float, default=2)
    parser.add_argument('--spatial_frequency', type=float, default=0.04)
    parser.add_argument('--init_screen_dur', type=float, default=0.5)
    parser.add_argument('--visual_flow_dur', type=float, default=2)
    parser.add_argument('--end_screen_dur', type=float, default=0)
    parser.add_argument('--contrast', type=float, default=0.8)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--no-reverse', dest='reverse', action='store_false')
    parser.set_defaults(reverse=False)
    parser.add_argument('--init_gray_screen', action='store_true')
    parser.add_argument('--no-init_gray_screen',
                        dest='init_gray_screen', action='store_false')
    parser.set_defaults(init_gray_screen=False)
    parser.add_argument('--end_gray_screen', action='store_true')
    parser.add_argument('--no-end_gray_screen',
                        dest='end_gray_screen', action='store_false')
    parser.set_defaults(end_gray_screen=False)
    parser.add_argument('--save_init_frame', action='store_true')
    parser.add_argument('--no-save_init_frame',
                        dest='save_init_frame', action='store_false')
    parser.set_defaults(save_init_frame=False)
    parser.add_argument('--save_stimulus_array', action='store_true')
    parser.add_argument('--no-save_stimulus_array',
                        dest='save_stimulus_array', action='store_false')
    parser.set_defaults(save_stimulus_array=False)
    parser.add_argument('--save_movie', action='store_true')
    parser.add_argument('--no-save_movie',
                        dest='save_movie', action='store_false')
    parser.set_defaults(save_movie=True)
    parser.add_argument('--create_lgn_firing_rates', action='store_true')
    parser.add_argument('--no-create_lgn_firing_rates',
                        dest='create_lgn_firing_rates', action='store_false')
    parser.set_defaults(create_lgn_firing_rates=True)

    flags = parser.parse_args()

    main(flags)
