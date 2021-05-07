cd ../../System/

python Main.py -i "ML(Dataset('stage_change_test', 3), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(Dataset('slow_curve', 3), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(Dataset('fast_curve', 3), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(Dataset('extreme_short', 3), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(Dataset('extreme_long', 3), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(Dataset('errors_throughout', 3), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(Dataset('errors_at_end', 3), use_multiprocessing=False)" -t -s -e

python Main.py -i "ML(Dataset('stage_change_test', 3), use_multiprocessing=False)" -l -o "training_strategy_name" "['slow_curve', 'fast_curve']"
python Main.py -i "ML(Dataset('stage_change_test', 3), use_multiprocessing=False)" -l -o "training_strategy_name" "['extreme_short', 'extreme_long']"
python Main.py -i "ML(Dataset('stage_change_test', 3), use_multiprocessing=False)" -l -o "training_strategy_name" "['errors_throughout', 'errors_at_end']"