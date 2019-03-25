#!/usr/bin/env python3

# PROGRAMMER: Diego da Costa Oliveira
# DATE CREATED: Mar, 25, 2019.
# REVISED DATE:
# PURPOSE: Uses a trained network to predict the class for an input image
# BASIC USAGE: python predict.py /path/to/image checkpoint
#
# Parameters:
#     1. Return top K most likely cases as --top_k with default value 5
#     2. Use a mapping of categories to real names as --category_names with default value 'cat_to_name.json'
#     3. Set GPU usage (CUDA) as --gpu


from helpers import get_input_args_predict, predict


# get input args
in_arg = get_input_args_predict()

# predict image
top_p_list, flower_names = predict(in_arg.path_to_image, in_arg.path_to_checkpoint, in_arg.top_k,
                                   in_arg.category_names, in_arg.gpu)

# print flower name and associated probability
print(f'Flower name and associated probability:\n')
for i in range(len(top_p_list)):
    print(f'{i+1}. {flower_names[i]} - {top_p_list[i]:.3f}')
