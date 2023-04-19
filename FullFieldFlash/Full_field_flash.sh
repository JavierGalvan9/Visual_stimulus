#! /bin/bash

run -c 1 -m 10 -t 2:00 -o Out/full_field_flash.out -e Error/field.err "python Full_field_flash.py --black_to_white --init_screen_dur 1 --flash_dur 1 --end_screen_dur 1 --no-save_movie --create_lgn_firing_rates"
