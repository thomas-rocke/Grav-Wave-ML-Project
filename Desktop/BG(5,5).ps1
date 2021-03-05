cd ../System/

python Main.py -i "ML(BasicGenerator(5, 5, 0.2, 0.4), use_multiprocessing=False)" -t -s -e 1000
python Main.py -i "ML(BasicGenerator(5, 5, 0.5, 1.0), use_multiprocessing=False)" -t -s -e 1000
python Main.py -i "ML(BasicGenerator(5, 5, 1.0, 2.0), use_multiprocessing=False)" -t -s -e 1000