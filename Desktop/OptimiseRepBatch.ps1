cd ../System/

python Main.py -i "ML(SuperpositionGenerator(3, batch_size=8, repeats=8, training_strategy_name='opti_test_strat'), use_multiprocessing=False)" -o "repeats" "[1, 2, 4, 8, 16]"
python Main.py -i "ML(SuperpositionGenerator(3, batch_size=8, repeats=8, training_strategy_name='opti_test_strat'), use_multiprocessing=False)" -o "batch_size" "[1, 2, 4, 8, 16]"