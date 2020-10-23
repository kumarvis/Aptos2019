import os
from pathlib import Path

class ImageClassificationConfig:
    ## Experiment_Params
    Image_Depth = 3
    Project_name = 'Aptos2019'
    Unstructured_Data_Path = ''
    Base_Data_Path = '../input/aptos2019-blindness-detection'
    Path_Curr_Dir = Path(__file__).parent.absolute()
    Path_Parent_Dir = str(Path(Path_Curr_Dir).parents[0])
    Gt_Path = os.path.join(Path_Parent_Dir, 'input', 'aptos2019-blindness-detection', 'train.csv')
    Img_Ext = 'png'
    Num_Classes = 5
    CheckPoints_Path = ''
    Mode = 'train'
    Validation_Fraction = 0.2
    Dataset_Shuffle = True
    Cyclic_LR = True
    No_System_Threads = 8
    Device = "cuda"
    Number_GPU = 1
    
    ## Hyper_Params 
    learning_rate = 0.00008
    batch_size_per_gpu = 32
    batch_size = batch_size_per_gpu * Number_GPU
    optimizer = 'ADAM'
    epochs = 25
    img_dim = 224
    img_channels = 3


ConfigObj = ImageClassificationConfig()
