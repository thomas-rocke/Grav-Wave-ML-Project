cd ../../System/

python Main.py -i "ML(SuperpositionGenerator(3, batch_size=64, repeats=64, training_strategy_name='opti_test_strat'), use_multiprocessing=False, optimiser='RMSprop', learning_rate=0.0001)" -o "repeats" "[16, 32, 64, 128, 256]"
python Main.py -i "ML(SuperpositionGenerator(3, batch_size=64, repeats=64, training_strategy_name='opti_test_strat'), use_multiprocessing=False, optimiser='RMSprop', learning_rate=0.0001)" -o "batch_size" "[16, 32, 64, 128, 256]"