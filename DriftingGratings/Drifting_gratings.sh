 #! /bin/bash


for orientation in 0 #45 90 135 180 225 270 315
do for TF in 2 #1 3 5 7 9 0 4 6 8 10
do run -c 1 -m 10 -t 2:00 -o Out/or-"$orientation"_TF-"$TF".out -e Error/or-"$orientation"_TF-"$TF".err "python Drifting_gratings.py --gratings_orientation $orientation --temporal_frequency $TF --contrast 0.8 --init_screen_dur 0.5 --visual_flow_dur 1 --end_screen_dur 1 --no-init_gray_screen --no-end_gray_screen --no-reverse --no-save-movie --create_lgn_firing_rates"
done
done
