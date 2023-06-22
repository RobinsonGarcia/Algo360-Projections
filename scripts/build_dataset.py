from pathlib import Path

import numpy as np
from PIL import Image

import argparse
import sys
import os

import pandas as pd
import multiprocessing as mp


sys.path.append('/nethome/algo360/mestrado/monocular-depth-estimation/Algo360Projections')
import algo360_projections as ap

parser = argparse.ArgumentParser(
                    prog='Dataset Builder',
                    description='Given an input path create projections of all images within and save them at an output root folder',
                    epilog='Developed by Robinson Garcia')

parser.add_argument('-i', '--input_folder',help='absolute path to a folder containing equirectangular images')      # option that takes a value
parser.add_argument('-o', '--output_folder',help='absolute path to an output folder that will store all projections')
parser.add_argument('-s','--save_png',action='store_true',help='bool to save sample images or not')

args = vars(parser.parse_args())


def mkdir(path):
    try:
        os.mkdir(path)
    except:
        print('dir already exists!')

    pass

def run(file):

    filename = file.split('/')[-1].split('.')[0]

    plataforma = file.split('/')[-2]

    output_folder = os.path.join(os.environ['OUTPUT_FOLDER'] ,plataforma)

    save_png = bool(os.environ['SAVE_PNG'])

    meta_csv = os.path.join(output_folder,'.metas/{}.csv'.format(filename))

    eq_img = ap.EquirectProjection.from_file(file)

    faces = eq_img.ico(2,random_samples=30)

    ixs = np.arange(len(faces))

    metas = []

    for ix in ixs:
        print(ix)

        face = faces[ix]
        angle_phi, angle_theta = faces.angles[:,ix]
        pos = faces.points[:,ix]

        saveas = os.path.join(output_folder,'{}_{:.2f}_{:.2f}.npz'.format(filename,angle_phi,angle_theta))

        meta = {'face':face , 
        'phi':angle_phi,
        'theta':angle_theta,
        'pos_x':pos[0],
        'pos_y':pos[1],
        'pos_z':pos[2],
        'face_filename':saveas,
        'eq_filename':file,
        'plataforma':plataforma,
        'eq_H':faces.eq_H,
        'eq_W':faces.eq_W}

        np.savez_compressed(saveas,**meta)

        if save_png:
            saveas_png = os.path.join(output_folder,'{}_{:.2f}.png'.format(filename,angle_phi))
            Image.fromarray(face).save(saveas_png)

        
        meta.pop('face')
        
        metas.append(meta)
        
        print(f'saving file {saveas}')


    faces = eq_img.cube()

    ixs = np.arange(len(faces))

    for ix in ixs:

        

        face = faces[ix]
        angle_phi, angle_theta = faces.angles[:,ix]
        pos = faces.points[:,ix]

        saveas = os.path.join(output_folder,'{}_cube_{:.2f}_{:.2f}.npz'.format(filename,angle_phi,angle_theta))

        meta = {'face':face , 
        'phi':angle_phi,
        'theta':angle_theta,
        'pos_x':pos[0],
        'pos_y':pos[1],
        'pos_z':pos[2], 
        'face_filename':saveas,
        'eq_filename':file, 
        'plataforma':plataforma,
        'eq_H':faces.eq_H,
        'eq_W':faces.eq_W}

        np.savez_compressed(saveas,**meta)

        if save_png:
            saveas_png = os.path.join(output_folder,'{}_cube_{:.2f}_{:.2f}.png'.format(filename,angle_phi,angle_theta))
            Image.fromarray(face).save(saveas_png)
        meta.pop('face')
        metas.append(meta)
        print(f'saving file {saveas}')  

    df = pd.DataFrame(metas)
    df.to_csv(meta_csv)

if __name__=="__main__":
    os.environ['OUTPUT_FOLDER'] = args['output_folder']
    os.environ['SAVE_PNG'] = 'True' if args['save_png'] else 'False'


    plataformas = os.listdir(args['input_folder'])
    for p in plataformas:
        mkdir(os.path.join( args['output_folder'],p))
        mkdir(os.path.join( args['output_folder'],p,'.metas'))

    files = [str(i) for i in Path(args['input_folder']).glob('**/*.JPG')]
    #files = [str(i) for i in Path(os.path.join(args['input_folder'],'P74')).glob('**/*.JPG')]
    N = len(files)
    print(f'found {N} files')


    cpu_count = mp.cpu_count()
    pool = mp.Pool(cpu_count)
    pool.map(run,files)
    pool.close()
        

