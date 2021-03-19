cd ../../System/

python Main.py -i "ML(SuperpositionGenerator(3, batch_size=8, repeats=8, training_strategy_name='opti_test_strat'), use_multiprocessing=False)" -o "learning_rate" "[round(0.1**n, n) for n in range(8)]"
python Main.py -i "ML(SuperpositionGenerator(3, batch_size=8, repeats=8, training_strategy_name='opti_test_strat'), use_multiprocessing=False)" -o "learning_rate" "learning_rate" "[round(0.0001 * n, 4) for n in range(1, 9)]"