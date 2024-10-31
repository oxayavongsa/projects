import sys
sys.path.append('../')
from pycore.tikzeng import *

# define your architecture
arch = [
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    # input layer with dimensions
    to_input('input.png', to="(-3,0,0)", width=10, height=12, name="input"),
    to_Conv(name='input_dim', s_filer=128, n_filer=150, offset="(0,0,0)", to="(input-east)", height=48, depth=56, width=1, caption="Input (128,150,1)"),
    
    # Conv2D -> MaxPooling2D -> Dropout
    to_Conv(name='conv2d_8', s_filer=126, n_filer=32, offset="(2,0,0)", to="(input_dim-east)", height=40, depth=48, width=3, caption="Conv2D 32"),
    to_Pool(name="max_pooling2d_8", offset="(2.5,0,0)", to="(conv2d_8-east)", height=30, depth=36, width=2, caption="MaxPool"),
    to_connection("input_dim", "conv2d_8"),
    to_Conv(name='dropout_12', s_filer=63, n_filer=32, offset="(2.75,0,0)", to="(max_pooling2d_8-east)", height=30, depth=36, width=1.5, caption="Dropout"),
    to_connection("max_pooling2d_8", "dropout_12"),
    
    # Conv2D -> MaxPooling2D -> Dropout
    to_Conv(name='conv2d_9', s_filer=61, n_filer=64, offset="(3,0,0)", to="(dropout_12-east)", height=30, depth=36, width=4, caption="Conv2D 64"),
    to_Pool(name="max_pooling2d_9", offset="(2.5,0,0)", to="(conv2d_9-east)", height=20, depth=24, width=2, caption="MaxPool"),
    to_connection("dropout_12", "conv2d_9"),
    to_connection("conv2d_9", "max_pooling2d_9"),
    to_Conv(name='dropout_13', s_filer=30, n_filer=64, offset="(2.75,0,0)", to="(max_pooling2d_9-east)", height=20, depth=24, width=1.5, caption="Dropout"),
    to_connection("max_pooling2d_9", "dropout_13"),
    
    # Flatten -> Dense -> Dropout -> Dense
    to_Conv(name='flatten_4', s_filer=69120, n_filer=128, offset="(3.5,0,0)", to="(dropout_13-east)", height=2, depth=2, width=12, caption="Flatten"),
    to_connection("dropout_13", "flatten_4"),
    to_Conv(name='dense_8', s_filer=128, n_filer=128, offset="(3,0,0)", to="(flatten_4-east)", height=8, depth=8, width=6, caption="Dense 128"),
    to_connection("flatten_4", "dense_8"),
    to_Conv(name='dropout_14', s_filer=128, n_filer=128, offset="(2.75,0,0)", to="(dense_8-east)", height=8, depth=8, width=2.5, caption="Dropout"),
    to_connection("dense_8", "dropout_14"),
    to_SoftMax(name='dense_9', s_filer=4, offset="(3,0,0)", to="(dropout_14-east)", width=2, height=3, depth=3, caption="Dense 4"),
    to_connection("dropout_14", "dense_9"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()
