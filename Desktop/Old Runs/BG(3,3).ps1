cd ../../System/

python Main.py -i "ML(BasicGenerator(3, 3, 0.2, 0.4), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(BasicGenerator(3, 3, 0.5, 1.0), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(BasicGenerator(3, 3, 1.0, 2.0), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(BasicGenerator(3, 3, 1.0, 0.0), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(BasicGenerator(3, 3, 0.0, 2.0), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(BasicGenerator(3, 9), use_multiprocessing=False)" -t -s -e