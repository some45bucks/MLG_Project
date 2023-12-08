python3 src/train.py.py --algo=APPO --env=doom_deadly_corridor --experiment=hallway_exp_gtest --save_every_sec=30 --use_rnn=True --use_geo=True --experiment_summaries_interval=500

python src/eval.py.py --algo=APPO --env=doom_deadly_corridor --experiment=hallway_exp_gtest --load_checkpoint_kind=best