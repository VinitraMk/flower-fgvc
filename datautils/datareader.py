import os
from scipy.io import loadmat

def get_img_filename(id):
    #print('id', id)
    nd = len(str(id))
    nz = 5 - nd
    fnum = "0" * nz + str(id)
    fname = f"image_{fnum}.jpg"
    
    return fname
    
def get_file_paths(data_dir):
    #img_dir = os.path.join(data_dir, 'jpg')
    #seg_mask_dir = os.path.join(data_dir, 'segmim')
    #labels_path = os.path.join(data_dir, 'imagelabels.mat')
    #labels_mat = loadmat(labels_path)
    ids_path = os.path.join(data_dir, 'setid.mat')
    ids_mat = loadmat(ids_path)
    #print('ids mat', ids_mat)
    train_ids = ids_mat['trnid'][0].tolist()
    test_ids = ids_mat['tstid'][0].tolist()
    val_ids = ids_mat['valid'][0].tolist()
    #print('id lens', len(train_ids), len(test_ids), len(val_ids))
    train_filenames = list(map(get_img_filename, train_ids))
    val_filenames = list(map(get_img_filename, val_ids))
    test_filenames = list(map(get_img_filename, test_ids))
    all_filenames = list(map(get_img_filename, list(range(8189))))
    
    return train_filenames, val_filenames, test_filenames, all_filenames