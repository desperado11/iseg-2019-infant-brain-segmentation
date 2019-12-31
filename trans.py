import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib


RAW_DATA_DIR = '/data2/lzh/3D-Unet-test/Training'
LABEL_DIR = '/data2/lzh/tfrecords_full'
PRED_DIR = '/data2/lzh/ca_results'
PRED_ID = 10# 1-10
PATCH_SIZE = 32
CHECKPOINT_NUM = 52800
OVERLAP_STEPSIZE = 8
SLICE_DEPTH = 150

def Visualize(label_dir, pred_dir, pred_id, patch_size, checkpoint_num,
		overlap_step):
    print('Loading predition...')
    pred_file = os.path.join(pred_dir,
                             'preds-%d-sub-%d-overlap-%d-patch-%d.npy' % \
                             (checkpoint_num, pred_id, overlap_step, patch_size))
    assert os.path.isfile(pred_file), \
        ('Run main.py --option=predict to generate the prediction results.')
    pred = np.load(pred_file)
    print('Check pred: ', pred.shape, np.max(pred))
    srcvol = nib.load('/data2/lzh/3D-Unet-test/Training/subject-10-label.hdr')
    nib.Nifti1Image(pred,srcvol.affine, srcvol.header).to_filename(
        '/data2/lzh/3D-Unet-test/nii/pred_id%d.nii' % pred_id)

if __name__ == '__main__':
	Visualize(
		label_dir=LABEL_DIR,
		pred_dir=PRED_DIR,
		pred_id=PRED_ID,
		patch_size=PATCH_SIZE,
		checkpoint_num=CHECKPOINT_NUM,
		overlap_step=OVERLAP_STEPSIZE)