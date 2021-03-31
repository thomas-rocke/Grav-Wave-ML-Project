cd ../../System/

python Main.py -i "ML(BasicGenerator(3, 3), use_multiprocessing=False)" -l -o "architecture" "['default', 'VGG16', 'VGG19']"
python Main.py -i "ML(Dataset('errors_at_end', 3), use_multiprocessing=False)" -l -o "architecture" "['default', 'VGG16', 'VGG19']"