# NN-visualizer

# example 1
python3 __main__.py --from-csv --dataset meta_model.csv --layers 1,2,32,32 --features Id --label SalePrice

# example 2
python3 __main__.py --dataset MOONS --transforms X1_POWER_2,SIN_X2 --layers 2,2,32,32
