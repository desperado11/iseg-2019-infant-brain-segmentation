import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib


RAW_DATA_DIR = '/data2/lzh/3D-Unet-test/Training'
# LABEL_DIR = '/data2/ql/3D-Unet-iseg2019/tfrecords_full'
PRED_DIR = '/data/lzh/final_result'
# PRED_ID = 39# 1-10
PATCH_SIZE = 32
CHECKPOINT_NUM = [22100,28000,28800,29000]
OVERLAP_STEPSIZE = 8
# SLICE_DEPTH = 150

def Visualize(pred_dir, pred_id, patch_size, overlap_step):
    print('start subject :',pred_id)
    print('Loading predition...')

    subject_dir = os.path.join(pred_dir,'subject%d'%(pred_id))
    print(subject_dir)
    npy=os.listdir(subject_dir+os.sep)
    print(len(npy))

    # for x in range(7):
    #     checkpoint_step = checkpoint_num[x]
    #     print(checkpoint_step)
    #
    #     pred_file[x] = os.path.join(pred_dir,'subject%d'%(pred_id),
    #                              'preds-%d-sub-%d-overlap-%d-patch-%d.npy' % \
    #                              (checkpoint_step, pred_id, overlap_step, patch_size))
    #     print('start processing',pred_file[x])
    #     assert os.path.isfile(pred_file[x]), \
    #         ('Run main.py --option=predict to generate the prediction results.')
    #     pred = np.load(pred_file[x])
    #     print('Check pred: ', pred.shape, np.max(pred))
    #npy
    pred_0 = np.load(os.path.join(subject_dir,npy[0]))
    pred_1 = np.load(os.path.join(subject_dir,npy[1]))
    pred_2 = np.load(os.path.join(subject_dir,npy[2]))
    pred_3 = np.load(os.path.join(subject_dir,npy[3]))
    # pred_4 = np.load(os.path.join(subject_dir,npy[4]))
    # pred_5 = np.load(os.path.join(subject_dir,npy[5]))
    # pred_6 = np.load(os.path.join(subject_dir,npy[6]))

    print('Check pred 0: ', pred_0.shape, np.max(pred_0),pred_0.dtype)
    print('Check pred 1: ', pred_1.shape, np.max(pred_1))
    print('Check pred 2: ', pred_2.shape, np.max(pred_2))
    print('Check pred 3: ', pred_3.shape, np.max(pred_3))
    # print('Check pred 4: ', pred_4.shape, np.max(pred_4))
    # print('Check pred 5: ', pred_5.shape, np.max(pred_5))
    # print('Check pred 6: ', pred_6.shape, np.max(pred_6))





    h, w ,d = pred_0.shape[0], pred_0.shape[1], pred_0.shape[2]
    vote_pred = np.zeros((h,w,d), dtype=np.int64)

    for y in range(h):
        for x in range(w):
            for z in range(d):
                pred_list = np.array([pred_0[y, x, z], pred_1[y, x, z], pred_2[y, x, z], pred_3[y, x, z]])
            # bin给出索引值在列表中出现的次数，argmax找出最大值(即预测结果最多的)的索引
                vote_pred[y, x, z] = np.argmax(np.bincount(pred_list))  # 先
    print('Check vote pred : ',vote_pred.shape,np.max(vote_pred),vote_pred.dtype)
    #
    #
    srcvol = nib.load('/data2/lzh/3D-Unet-test/Training/subject-%d-T1.hdr'%pred_id)
    nib.Nifti1Image(vote_pred,srcvol.affine, srcvol.header).to_filename(
        '/data/lzh/vote/subject-%d-label.nii' % pred_id)

if __name__ == '__main__':


    for i in range (39,40):

        Visualize(pred_dir=PRED_DIR,pred_id=i,patch_size=PATCH_SIZE,overlap_step=OVERLAP_STEPSIZE)