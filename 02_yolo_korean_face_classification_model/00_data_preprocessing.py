import glob
import os
import sys
from os import path
import shutil
import argparse
from tqdm import tqdm

def error(msg):
    print('Error: ' + msg)
    exit(1)

def create_kface(target_dir, kface_image_dir, light, resolution, bbox):#, resize_val, shuffle):
    print('Loading images from "%s"' % kface_image_dir)
    resolution_nm = resolution + '_Resolution'

    sessions = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006']
    emotions = ['E01', 'E02', 'E03']
    cameras = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13']

    # 대상자 id폴더 리스트
    pth_subj_data = []
    pth_subj_data = sorted(glob.glob(os.path.join(kface_image_dir, resolution_nm, '*')))
    len_subj_data = len(pth_subj_data)
    subjid = []
    for idx in range(len_subj_data):
        subjid.append(pth_subj_data[idx].split('\\')[-1])

    for session in tqdm(sessions, desc='Accessory'):
        for emotion in tqdm(emotions, desc='Emotion'):
            for camera in tqdm(cameras, desc='Camera'):
                filenames = list()
                if bbox == 'Y':
                    txtnames = list()
                for idx in range(0, len_subj_data):
                    tmp_nm = []
                    tmp_nm = glob.glob(os.path.join(pth_subj_data[idx], session, light, emotion, camera+'.jpg'))
                    filenames.append(tmp_nm)
                    
                    if bbox == 'Y' and session == 'S001':
                        txt_nm = []
                        txt_nm = glob.glob(os.path.join(pth_subj_data[idx], session, light, emotion, camera+'.txt'))
                        txtnames.append(txt_nm)

                if len(filenames) == 0:
                    error('No input images found')
                for idx in range(0, len_subj_data):
                    to_dir = target_dir+'\\'+kface_image_dir+'\\'+resolution_nm+'\\'+session+'_'+emotion
                    if not path.isdir(to_dir):
                        os.makedirs(to_dir)
                    if not path.exists(to_dir+'\\'+subjid[idx]+'_'+camera+'.jpg'):
                        shutil.copy(filenames[idx][0], to_dir+'\\'+subjid[idx]+'_'+camera+'.jpg')
                    if bbox == 'Y' and session == 'S001':
                        if not path.exists(to_dir+'\\'+subjid[idx]+'_'+camera+'.txt'):
                            shutil.copy(txtnames[idx][0], to_dir+'\\'+subjid[idx]+'_'+camera+'.txt')

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for creating Progressive GAN datasets.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)
        
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(    'create_kface', 'Create dataset from a directory full of K-Face images.',
                                            'create_kface datasets/my/image/dir my/image/dir')
    p.add_argument(     'target_dir',     help='New tfrecord dataset directory to be created')
    p.add_argument(     'kface_image_dir',  help='Directory containing the K-Face images')
    # p.add_argument(     '--session',        help='Select one session type [S001~S006] (default: S001)', type=str, default='S001')
    p.add_argument(     '--light',          help='Select one light type [L1~L30] (default: L1)', type=str, default='L1')
    # p.add_argument(     '--emotion',        help='Select one emotion type [E01, E02, E03] (default: E01)', type=str, default='E01')
    # p.add_argument(     '--camera',         help='Select one camera(degree) type [C1~C20] (default: C7)', type=str, default='C7')
    p.add_argument(     '--resolution',     help='Select resolution type. Input one of three types: High, Middle, Low (default: High)', type=str, default='High')
    # p.add_argument(     '--resize_val',     help='Input resizing value (should be power of 2). (default: 512)', type=int, default=512)
    p.add_argument(     '--bbox',           help='Check whether using bbox info or not (Y/N). (default: Y)', type=str, default='Y')
    # p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)


    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

if __name__ == "__main__":
    execute_cmdline(sys.argv)