cd ../../System/

python Main.py -i "ML(SuperpositionGenerator(3, batch_size=8, repeats=8, training_strategy_name='opti_test_strat'), use_multiprocessing=False, optimiser='RMSprop', learning_rate=0.0001)" -o "training_strategy_name" "['class_then_vary', 'variance_throughout']"
