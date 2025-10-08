#!/bin/zsh

python make_obs_sim.py three_pts 1 0.55 --alpha=-8.0 --edgesfile=three_pts.edges
python make_obs_sim.py three_pts 1 0.4 --alpha=-8.0 --edgesfile=three_pts.edges
python make_obs_sim.py three_pts 1 0.4 --alpha=-5.0 --edgesfile=three_pts.edges
python make_obs_sim.py three_pts 1 0.63 --edgesfile=three_pts.edges
python make_obs_sim.py three_pts 1 0.4 --edgesfile=three_pts.edges
python make_obs_sim.py three_pts 50 0.4 --edgesfile=three_pts.edges

python make_obs_sim.py three_pts 1 0.55 --alpha=-8.0 --noerr --edgesfile=three_pts.edges
python make_obs_sim.py three_pts 1 0.4 --alpha=-8.0 --noerr --edgesfile=three_pts.edges
python make_obs_sim.py three_pts 1 0.4 --alpha=-5.0 --noerr --edgesfile=three_pts.edges
python make_obs_sim.py three_pts 1 0.63 --noerr --edgesfile=three_pts.edges
python make_obs_sim.py three_pts 1 0.4 --noerr  --edgesfile=three_pts.edges
python make_obs_sim.py three_pts 50 0.4 --noerr --edgesfile=three_pts.edges

python make_obs_sim.py two_pts 1 0.55 --alpha=-8.0 --edgesfile=three_pts.edges
python make_obs_sim.py two_pts 1 0.4 --alpha=-8.0 --edgesfile=three_pts.edges
python make_obs_sim.py two_pts 1 0.4 --alpha=-5.0 --edgesfile=three_pts.edges
python make_obs_sim.py two_pts 1 0.63 --edgesfile=three_pts.edges
python make_obs_sim.py two_pts 1 0.4 --edgesfile=three_pts.edges
python make_obs_sim.py two_pts 50 0.4 --edgesfile=three_pts.edges

python make_obs_sim.py two_pts 1 0.55 --alpha=-8.0 --noerr --edgesfile=three_pts.edges
python make_obs_sim.py two_pts 1 0.4 --alpha=-8.0 --noerr --edgesfile=three_pts.edges
python make_obs_sim.py two_pts 1 0.4 --alpha=-5.0 --noerr --edgesfile=three_pts.edges
python make_obs_sim.py two_pts 1 0.63 --noerr --edgesfile=three_pts.edges
python make_obs_sim.py two_pts 1 0.4 --noerr  --edgesfile=three_pts.edges
python make_obs_sim.py two_pts 50 0.4 --noerr --edgesfile=three_pts.edges
