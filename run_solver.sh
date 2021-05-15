#install dependencies
conda install opencv --y
conda install pytorch --y
conda install scikit-learn --y
conda install torchvision --y
conda install tqdm --y
conda install numpy --y

python3 main.py --train_data_dir $1 --test_data_dir $2 --sample_file $3 --output_file $4