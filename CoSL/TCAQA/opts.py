# directory containing frames
frames_dir = ''
# directory containing labels and annotations
info_dir = ''


i3d_pretrained_path = ''

# num of frames in a single video
num_frames = 103

# beginning frames of the 10 segments
segment_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 87]

# input data dims;
C, H, W = 3,224,224
# image resizing dims;
input_resize = 455, 256

# statistics of dataset
label_max = 104.5
label_min = 0.
judge_max = 10.
judge_min = 0.

# output dimension of I3D backbone
feature_dim = 1024


output_dim = {'USDL-CoSL':101, 'MUSDL-CoSL': 21}

# num of judges in MUSDL-CoSL method
num_judges = 5
