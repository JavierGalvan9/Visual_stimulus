import numpy as np

def make_drifting_grating_stimulus(row_size=120, col_size=240, moving_flag=True, image_duration=100, cpd = 0.05,
            temporal_f = 2, theta = 45, phase = None, contrast = 1.0):
    # parameters from Allen's code
    '''
    Create the grating movie with the desired parameters
    :param t_min: start time in seconds
    :param t_max: end time in seconds
    :param cpd: cycles per degree
    :param temporal_f: in Hz
    :param theta: orientation angle
    :return: Movie object of grating with desired parameters
    '''
    row_size = row_size*2 # somehow, Franz's code only accept larger size; thus, i did the mulitplication
    col_size = col_size*2
    frame_rate = 1000 # Hz
    t_min = 0
    t_max = image_duration/1000
    if phase is None:
        phase = np.random.rand(1)*180

    assert contrast <= 1, "Contrast must be <= 1"
    assert contrast > 0, "Contrast must be > 0"

    physical_spacing = 1.# tf version lgn model need this to keep true cpd; / (float(cpd) * 10)    #To make sure no aliasing occurs
    row_range  = np.linspace(0, row_size, int(row_size / physical_spacing), endpoint = True)
    col_range  = np.linspace(0, col_size, int(col_size / physical_spacing), endpoint = True)
    numberFramesNeeded = int(round(frame_rate * t_max))
    time_range = np.linspace(0, t_max, numberFramesNeeded, endpoint=True)   ### this was a bug... instead of zero it was gray_screen and so time was stretched! Fixed on Jan 11, 2018

    tt, yy, xx = np.meshgrid(time_range, row_range, col_range, indexing='ij')

    thetaRad = np.pi*(180-theta)/180.   #Add negative here to match brain observatory angles!
    phaseRad = np.pi*(180-phase)/180.
    xy = xx * np.cos(thetaRad) + yy * np.sin(thetaRad)
    data = contrast*np.sin(2*np.pi*(cpd * xy + temporal_f *tt) + phaseRad)
    if moving_flag:
        return data.astype(np.float32)
    else:
        return np.tile(data[0].astype(np.float32)[None,...],(image_duration,1,1))