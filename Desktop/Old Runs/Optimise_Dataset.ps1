cd ../../System/

python Main.py -i "ML(BasicGenerator(3, 3), use_multiprocessing=False)" -o "starting_stage" "[1, 2]"
python Main.py -i "ML(BasicGenerator(3, 3), use_multiprocessing=False)" -o "batch_size" "[2**n for n in range(9)]"
python Main.py -i "ML(BasicGenerator(3, 3), use_multiprocessing=False)" -o "resolution" "[8, 16, 32, 64, 128]"
python Main.py -i "ML(BasicGenerator(3, 3), use_multiprocessing=False)" -o "repeats" "[2**n for n in range(1, 9)]"

python Main.py -i "ML(SuperpositionGenerator('default', 3), use_multiprocessing=False)" -o "batch_size" "[2**n for n in range(7)]"
python Main.py -i "ML(SuperpositionGenerator('default', 3), use_multiprocessing=False)" -o "resolution" "[8, 16, 32, 64, 128]"
python Main.py -i "ML(SuperpositionGenerator('default', 3), use_multiprocessing=False)" -o "resolution" "[64, 128, 256]"
python Main.py -i "ML(SuperpositionGenerator('default', 3), use_multiprocessing=False)" -o "repeats" "[2*n for n in range(1, 4)]"
python Main.py -i "ML(SuperpositionGenerator('default', 3), use_multiprocessing=False)" -o "repeats" "[3, 4, 5]"