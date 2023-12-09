You need to install VizDoom. You can use the repo: https://github.com/Farama-Foundation/ViZDoom, or just `pip install vizdoom`. You need to run this on Linux or WSL or something like that. It will not run otherwise!

I have all the modified code in my repo so you shouln't need to also install sample factory 

You can test the training but it will be slow, but shouldn't error out. I recommend running the eval ones so you can see whats happening, and they are much faster

Commands for running and verifiying code:

**With Auxiliary**
- Train Deadly Corridor Auxiliary:
`python3 src/train.py --algo=APPO --env=doom_deadly_corridor --experiment=hallway_exp_gtest --save_every_sec=30 --use_rnn=True --use_geo=True --experiment_summaries_interval=500`

- Eval Deadly Corridor Auxiliary:
`python3 src/eval.py --algo=APPO --env=doom_deadly_corridor --experiment=hallway_exp_gtest --load_checkpoint_kind=best`

- Train My Way Home Auxiliary:
`python3 src/train.py --algo=APPO --env=doom_my_way_home --experiment=my_way_home_gtest --save_every_sec=30 --use_rnn=True --use_geo=True --experiment_summaries_interval=500`

- Eval My Way Home Auxiliary:
`python3 src/eval.py --algo=APPO --env=doom_my_way_home --experiment=my_way_home_gtest --load_checkpoint_kind=best`

**Base Line**
- Train Deadly Corridor Normal:
`python3 src/train.py --algo=APPO --env=doom_deadly_corridor --experiment=hallway_exp_normal --save_every_sec=30 --use_rnn=True --use_geo=True --experiment_summaries_interval=500`

- Eval Deadly Corridor Normal:
`python3 src/eval.py --algo=APPO --env=doom_deadly_corridor --experiment=hallway_exp_normal --load_checkpoint_kind=best`

- Train My Way Home Normal:
`python3 src/train.py --algo=APPO --env=doom_my_way_home --experiment=my_way_home_ntest --save_every_sec=30 --use_rnn=True --use_geo=True --experiment_summaries_interval=500`

- Eval My Way Home Normal:
`python3 src/eval.py --algo=APPO --env=doom_my_way_home --experiment=my_way_home_ntest --load_checkpoint_kind=best`