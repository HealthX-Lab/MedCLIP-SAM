import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from skimage.segmentation import relabel_sequential
import os
import argparse

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess_crf(args):
    files = os.listdir(args.input_path)

    if (not os.path.exists(args.output_path)):
        os.makedirs(args.output_path)

    for file in files:

        img = cv2.imread(args.input_path+'/'+file, 1)
        annos = cv2.imread(args.sal_path+'/'+file, 0)
        labels = relabel_sequential(cv2.imread(args.sal_path+'/'+file, 0))[0].flatten()
        output = args.output_path+'/'+file

        # Setup the CRF model
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], args.m)

        anno_norm = annos / 255.
        n_energy = -np.log((1.0 - anno_norm + args.epsilon)) / (args.tau * sigmoid(1 - anno_norm))
        p_energy = -np.log(anno_norm + args.epsilon) / (args.tau * sigmoid(anno_norm))

        U = np.zeros((args.m, img.shape[0] * img.shape[1]), dtype='float32')
        U[0, :] = n_energy.flatten()
        U[1, :] = p_energy.flatten()

        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=args.gaussian_sxy, compat=3)
        d.addPairwiseBilateral(sxy=args.bilateral_sxy, srgb=args.bilateral_srgb, rgbim=img, compat=5)

        # Do the inference
        Q = d.inference(1)
        map = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

        # Save the output as image
        map *= 255
        cv2.imwrite(output, map.astype('uint8'))

    print("CRF postprocessing done!")
        


def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gaussian-sxy', type=int, default=3,
                        help="Gaussian sxy value for CRF")
    parser.add_argument('--bilateral-sxy', type=int, default=60,
                        help="Bilateral sxy value for CRF")
    parser.add_argument('--bilateral-srgb', type=int, default=5,
                        help="Bilateral srgb value for CRF")
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help="Epsilon value for CRF")
    parser.add_argument('--m', type=int, default=2,
                        help="Number of classes in the saliency map")
    parser.add_argument('--tau', type=float, default=1.05,
                        help="Tau value for CRF")
    parser.add_argument('--input-path', type=str, default='images',
                        help="Path to the images")
    parser.add_argument('--sal-path', type=str, default='cams',
                        help="Path to the saliency maps")
    parser.add_argument('--output-path', type=str, default='output',
                        help="Output path of CRF postprocessed samples")

    return parser.parse_args()
if __name__ == '__main__':
    args = get_parser()
    postprocess_crf(args)

