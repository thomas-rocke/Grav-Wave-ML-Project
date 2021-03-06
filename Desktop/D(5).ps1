cd ../System/

python Main.py -i "ML(Dataset('stage_change_test', 5), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(Dataset('slow_curve', 5), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(Dataset('fast_curve', 5), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(Dataset('errors_throughout', 5), use_multiprocessing=False)" -t -s -e
python Main.py -i "ML(Dataset('errors_at_end', 5), use_multiprocessing=False)" -t -s -e

python Main.py -i "ML(Dataset('stage_change_test', 5), use_multiprocessing=False)" -l -o "training_strategy_name" "['stage_change_test', 'slow_curve', 'fast_curve', 'errors_throughout', 'errors_at_end']"