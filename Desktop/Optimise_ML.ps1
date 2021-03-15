cd ../System/

python Main.py -i "ML(BasicGenerator(3, 3), use_multiprocessing=False)" -o "optimiser" "['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']"
python Main.py -i "ML(BasicGenerator(3, 3), use_multiprocessing=False)" -o "learning_rate" "[round(0.1**n, n) for n in range(8)]"
python Main.py -i "ML(BasicGenerator(3, 3), use_multiprocessing=False)" -o "learning_rate" "[round(0.0001 * n, 4) for n in range(1, 9)]"

python Main.py -i "ML(Dataset('errors_at_end', 3), use_multiprocessing=False)" -o "optimiser" "['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']"
python Main.py -i "ML(Dataset('errors_at_end', 3), use_multiprocessing=False)" -o "learning_rate" "[round(0.1**n, n) for n in range(8)]"
python Main.py -i "ML(Dataset('errors_at_end', 3), use_multiprocessing=False)" -o "learning_rate" "[round(0.0001 * n, 4) for n in range(1, 9)]"