cd ../System/

python Main.py -i "ML(Dataset('errors_at_end', 5), use_multiprocessing=False)" -o "optimiser" "['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']"
python Main.py -i "ML(Dataset('errors_at_end', 5), use_multiprocessing=False)" -o "learning_rate" "[round(0.1**n, n) for n in range(8)]"
python Main.py -i "ML(Dataset('errors_at_end', 5), use_multiprocessing=False)" -o "learning_rate" "[round(0.0001 * n, 4) for n in range(1, 9)]"

python Main.py -i "ML(Dataset('errors_at_end', 5), use_multiprocessing=False)" -o "batch_size" "[2**n for n in range(7)]"
python Main.py -i "ML(Dataset('errors_at_end', 5), use_multiprocessing=False)" -o "resolution" "[8, 16, 32, 64, 128]"
python Main.py -i "ML(Dataset('errors_at_end', 5), use_multiprocessing=False)" -o "resolution" "[64, 128, 256]"
python Main.py -i "ML(Dataset('errors_at_end', 5), use_multiprocessing=False)" -o "repeats" "[2*n for n in range(1, 4)]"
python Main.py -i "ML(Dataset('errors_at_end', 5), use_multiprocessing=False)" -o "repeats" "[3, 4, 5]"

python Main.py -i "ML(Dataset('errors_at_end', 5), use_multiprocessing=False)" -o "noise_variation" "[round(0.1 * n, 1) for n in range(11)]"
python Main.py -i "ML(Dataset('errors_at_end', 5), use_multiprocessing=False)" -o "amplitude_variation" "[round(0.1 * n, 1) for n in range(11)]"
python Main.py -i "ML(Dataset('errors_at_end', 5), use_multiprocessing=False)" -o "phase_variation" "[round(0.2 * n, 1) for n in range(11)]"