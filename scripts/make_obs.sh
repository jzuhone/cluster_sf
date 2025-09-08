#!/bin/zsh

python make_obs_sim.py two_pts 1 0.55 --alpha=-8.0
python make_obs_sim.py two_pts 1 0.4 --alpha=-8.0
python make_obs_sim.py two_pts 1 0.4 --alpha=-5.0
python make_obs_sim.py two_pts 1 0.63
python make_obs_sim.py two_pts 1 0.4
python make_obs_sim.py two_pts 50 0.4

python make_obs_sim.py two_pts 1 0.55 --alpha=-8.0 --noerr
python make_obs_sim.py two_pts 1 0.4 --alpha=-8.0 --noerr
python make_obs_sim.py two_pts 1 0.4 --alpha=-5.0 --noerr
python make_obs_sim.py two_pts 1 0.63 --noerr
python make_obs_sim.py two_pts 1 0.4 --noerr
python make_obs_sim.py two_pts 50 0.4 --noerr
