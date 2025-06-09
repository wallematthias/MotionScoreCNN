# === Standard Library ===
import argparse
import os
import warnings
from glob import glob
from pathlib import Path

# === Environment setup ===
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# === Third-Party ===
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import gaussian
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# === Local ===
from aim import aim



def pad_to_square(image):
    """
        Pads a 2D image to make it square by adding black borders.

        Parameters
        ----------
        image : np.ndarray
            Input 2D image as a NumPy array with shape (H, W) or (H, W, C).

        Returns
        -------
        np.ndarray
            Square-padded image with equal height and width.
        """
    ...
    height, width = image.shape[:2]
    size = max(height, width)
    top_pad = (size - height) // 2
    bottom_pad = size - height - top_pad
    left_pad = (size - width) // 2
    right_pad = size - width - left_pad
    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)
    return padded_image

def preprocess_image(image):
    """
        Pads and resizes an image to 512x512 with normalized intensity.

        Parameters
        ----------
        image : np.ndarray
            Raw 2D image slice as a NumPy array.

        Returns
        -------
        np.ndarray
            Preprocessed image array suitable for DNN input (shape: 512x512x1).
    """
    # Pad the image to square dimensions
    padded_image = pad_to_square(image)

    # Normalize and convert to CV_8U dtype
    normalized_image = cv2.normalize(padded_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert to PIL Image
    pil_image = Image.fromarray(normalized_image)

    # Resize the image to [512, 512]
    resized_image = pil_image.resize((512, 512))

    return img_to_array(resized_image)

def plot_slice(data, slices, scores):
    """
        Plots selected slices from a 3D image volume with their corresponding scores.

        Parameters
        ----------
        data : np.ndarray
            3D image volume of shape (H, W, D).
        slices : list of int
            Indices of the slices to display.
        scores : list of int or float
            Motion scores for each slice.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure.
        axes : list of matplotlib.axes.Axes
            Axes for each subplot.
    """
    
    n_slices = len(slices) + 1
    # Compute the optimal figure width based on the size of the data array
    data_shape = np.rot90(data[:, :, slices[0]]).shape
    fsize = 3
    fig_width = max(1, n_slices * (data_shape[1] / data_shape[0]) * fsize)
    fig, axes = plt.subplots(1, n_slices, figsize=(fig_width, fsize), facecolor='white')
    for ax, s, score in zip(axes.flatten()[::-1], slices, scores):
        img = ax.imshow(np.rot90(data[:, :, s]))
        img.set_cmap('gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('# {} (grade {})'.format(s, score), fontsize=10)
    fig.subplots_adjust(wspace=0.05)
    return fig, axes

def get_indices(scores, confidences):
    """
        Selects the lowest, median, and highest scoring slices based on confidence.

        Parameters
        ----------
        scores : list or np.ndarray
            Predicted scores per slice.
        confidences : list or np.ndarray
            Confidence values associated with each score.

        Returns
        -------
        indices : list of int
            Indices of selected slices.
        scores : list of int
            Corresponding scores for selected slices.
    """


    score_dict = {}
    for i, score in enumerate(scores):
        if score not in score_dict:
            score_dict[score] = (i, score, confidences[i])
        else:
            if confidences[i] > score_dict[score][2]:
                score_dict[score] = (i, score, confidences[i])

    sorted_scores = sorted(score_dict.keys())

    lowest = sorted_scores[0]
    highest = sorted_scores[-1]
    if len(sorted_scores) % 2 == 0:
        median = sorted_scores[len(sorted_scores) // 2 - 1]
    else:
        median = sorted_scores[len(sorted_scores) // 2]

    indices = [score_dict[lowest][0], score_dict[median][0], score_dict[highest][0]]
    scores = [score_dict[lowest][1], score_dict[median][1], score_dict[highest][1]]

    if len(sorted_scores) < 3:
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k])
        indices = [sorted_indices[0], sorted_indices[len(sorted_indices) // 2], sorted_indices[-1]]
        scores = [scores[sorted_indices[0]], scores[len(sorted_indices) // 2], scores[sorted_indices[-1]]]

    return indices, scores

def automatic_motion_score(im_raw, outpath=None, stackheight=168):
    """
        Automatically computes motion scores from a 3D image using an ensemble of deep neural networks.

        The image is smoothed, preprocessed, and each slice is scored using a voting-based ensemble
        of trained CNN models. Slice-level scores are aggregated into stack-level motion scores,
        and a visualization is generated and saved as a PNG.

        Parameters
        ----------
        im_raw : np.ndarray
            The raw 3D image volume with shape (H, W, D), typically from an AIM file.
        
        outpath : str or None, optional
            The output file path prefix for saving the motion score plot. If None, defaults to the
            directory of the input image.

        stackheight : int, default=168
            Number of slices per stack for computing stack-level scores. Will be reduced to fit if
            the image depth is smaller.

        Returns
        -------
        tuple
            If multiple stacks are present:
                - scores : np.ndarray of shape (N+1,)
                    Stack-level motion scores followed by the mean motion score.
                - confidences : np.ndarray of shape (N+1,)
                    Average confidence values for each stack and overall.

            If only one stack:
                - score : float
                    Mean motion score.
                - confidence : float
                    Average confidence of all slices.

        Notes
        -----
        - Trained models are loaded from `models/*.h5` in the script directory.
        - The resulting plot is saved as `{outpath}_{score}_{confidence}_motion.png`.
        - Confidence is computed as the average of maximum softmax probabilities across all models.
    """
    if stackheight>im_raw.shape[2]:
        print(f'Stackheight input {stackheight} larger than image {im_raw.shape[2]} reducing stackheight')
        stackheight = im_raw.shape[2]

    model_paths = sorted(glob(os.path.join(
            os.path.dirname(
            Path(__file__)),
            'models','*.h5')))
    DNN_list = [load_model(model_path, custom_objects={'LeakyReLU': LeakyReLU()}) for model_path in model_paths]

    for i, DNN_model in enumerate(DNN_list):
        DNN_model._name = "model" + str(i)

    im_filtered = gaussian(im_raw, sigma=0.8, truncate=1.25)

    model_number = 10


    full_scan = np.asarray([preprocess_image(im_filtered[:, :, i]) for i in range(0, im_filtered.shape[2])])
    

    result = np.zeros((len(DNN_list), full_scan.shape[0], 5))
    for i in range(len(DNN_list)):
        result[i] = DNN_list[i].predict(full_scan / 255)

    all_result_binary = np.zeros(result.shape)
    all_result_matrix = np.zeros(result.shape[:2])

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            max_ = max(result[i, j])
            all_result_binary[i, j] = np.floor(result[i, j] / max_)
            all_result_matrix[i, j] = np.where(all_result_binary[i, j] == 1)[0][0]

    votes = []
    for i in range(all_result_matrix.shape[1]):
        bins = np.bincount(all_result_matrix[:, i].astype('int8'), minlength=5)
        votes.append(bins)
    votes = np.array(votes) / model_number

    category_colors_neg = plt.get_cmap('RdBu')(np.linspace(0.75, 0.95, 3))
    category_colors_pos = plt.get_cmap('RdBu')(np.linspace(0.05, 0.15, 2))

    class_1 = votes[:, 0]
    class_2 = votes[:, 1]
    class_3 = votes[:, 2]
    class_4 = votes[:, 3]
    class_5 = votes[:, 4]

    mscore = np.round(np.mean(np.argmax(votes, axis=1) + 1), 0)
    mscorevalue = np.round(np.mean(np.max(votes, axis=1)), 2)
    score = np.round(np.mean(np.reshape(np.argmax(votes, axis=1) + 1, (int(full_scan.shape[0] / stackheight), stackheight)), axis=1), 0)
    scorevalue = np.round(np.mean(np.reshape(np.max(votes, axis=1), (int(full_scan.shape[0] / stackheight), stackheight)), axis=1), 2)

    slice_score = np.argmax(votes, axis=1) + 1
    slice_conf = np.max(votes, axis=1)

    data = im_raw
    slices, scores = get_indices(slice_score, slice_conf)
    fig, axes = plot_slice(data, slices, scores)
    ax = axes[0]

    labels = range(0, full_scan.shape[0])

    ax.barh(y=labels, width=class_1, height=1, left=-(class_1 + class_2 + class_3), label='score 1', color=category_colors_neg[2])
    ax.barh(y=labels, width=class_2, height=1, left=-np.add(class_2, class_3), label='score 2', color=category_colors_neg[1])
    ax.barh(y=labels, width=class_3, height=1, left=-class_3, label='score 3', color=category_colors_neg[0])
    ax.barh(y=labels, width=class_4, height=1, label='score 4', color=category_colors_pos[1])
    ax.barh(y=labels, width=class_5, height=1, left=class_4, label='score 5', color=category_colors_pos[0])

    fig.legend(fontsize=10, loc='lower center', ncol=5)

    ax.set_ylim(0, full_scan.shape[0])
    ax.set_xlim(-1, 1)
    ax.set_ylabel('Slice #', fontsize=10, labelpad=10)
    ax.set_xlabel('')
    fig.suptitle('{} stacks ({} slices) Score {} ({})'.format(int(full_scan.shape[0] / stackheight), stackheight, mscore, mscorevalue), fontsize=12, y=0.95)
    ax.grid('on', axis='y', linewidth=0.5)

    fig.subplots_adjust(bottom=0.2)

    if outpath is None:
        outpath = os.path.dirname(path)
    else:
        outpath = os.path.abspath(outpath)

    for i, (y, s, val) in enumerate(zip(np.linspace(0, full_scan.shape[0] - stackheight, (int(full_scan.shape[0] / stackheight))), score, scorevalue)):
        if i > 0:
            ax.axhline(y, linestyle='--')
        t = ax.text(-0.9, y + stackheight / 2, 'Score {} ({})'.format(s, val), fontsize=8)
        t.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='white'))

    # Convert mscorevalue to a percentage
    mscore_percentage = int(mscorevalue * 100)
    mscore_int = int(mscore)

    # Create the filename using f-string formatting
    filename = f'{outpath}_{mscore_int}_{mscore_percentage}_motion.png'
        
    plt.savefig(filename, transparent=False)
    plt.close(fig)
    if len(score) > 1:
        return np.append(score, mscore), np.append(scorevalue, mscorevalue)
    else:
        return mscore, mscorevalue

def grade_images(file_paths, stackheight, outpath):
    """
        Applies automatic motion scoring to a list of AIM images and saves the results.

        Each AIM file is loaded, scored using `automatic_motion_score`, and the result is
        saved as a plot. Scores are printed for each file.

        Parameters
        ----------
        file_paths : list of str
            List of paths to `.AIM` image files.

        stackheight : int
            Number of slices per stack for motion scoring.

        outpath : str
            Directory where output images and results will be saved.
    """
    exclude_keywords = ['mask', 'trab', 'cort']
    paths = [p for p in file_paths if not any(k in os.path.basename(p).lower() for k in exclude_keywords)]

    for path in paths:
        file = aim.load_aim(path)
        name = os.path.basename(path).split('.')[0]
        mscore, mscorevalue = automatic_motion_score(
            file.data, outpath=os.path.join(outpath, name), stackheight=stackheight)
        print(f'Motion Score {name}: {mscore}')

def confirm_images(image_files, confidence_threshold, output_path):
    """
        Displays and manually confirms motion scores for PNG images, and records grading accuracy.

        Images are expected to be named using the convention: `name_score_confidence_motion.png`.
        If the confidence is below the threshold, the image is shown for manual confirmation.
        Results are saved to a CSV file and overall accuracy is reported.

        Parameters
        ----------
        image_files : list of str
            List of paths to motion score PNG files (e.g., *_motion.png).

        confidence_threshold : int
            Minimum confidence (%) above which the score is automatically accepted.

        output_path : str
            Path to the CSV file where results will be saved.

        Notes
        -----
        - Uses `matplotlib` to display images for manual review.
        - Prompts for user input in the terminal if confidence is low.
        - Output CSV includes filename, manual and automatic grades, and confidence.
    """
    data = []
    total_images = len(image_files)
    graded_images = 0

    for image_path in image_files:
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        file_parts = file_name.split('_')

        if len(file_parts) >= 3:
            motion_score_default = int(file_parts[-3])  
            confidence_default = int(file_parts[-2])
            filename = '_'.join(file_parts[:-3]) 
        else:
            print(f"Skipping file '{image_path}' due to unexpected filename format.")
            continue

        if confidence_default < confidence_threshold:
            graded_images += 1
            print(f"Graded {graded_images}/{total_images} ({image_path})")

            # Load and display the image using matplotlib
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{filename} (confidence: {confidence_default}%)")
            plt.tight_layout()
            plt.pause(0.001)
            plt.show(block=False)

            # Prompt for user input while image is shown
            motion_score = input(f"Enter your assessment for the motion score [{motion_score_default}]: ")
            plt.close()

            if not motion_score:
                motion_score = motion_score_default
        else:
            motion_score = motion_score_default
            graded_images += 1

        data.append({
            'filename': filename,
            'manual_grade': int(motion_score),
            'automatic_grade': int(motion_score_default),
            'confidence': int(confidence_default)
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

    correct = (df['manual_grade'] == df['automatic_grade']).sum()
    total = len(df)
    accuracy = correct / total if total > 0 else 0.0

    print(f"Grading complete. Saved to '{output_path}'. Accuracy: {accuracy:.2%}")
    

def parse_args():
    """
        Parses command-line arguments for grading or confirming motion scores.

        Returns
        -------
        argparse.Namespace
            Parsed arguments including mode, input files, output path, and additional options.

        Modes
        -----
        - grade: Automatically scores a list of AIM files.
            --input <*.AIM>
            --stackheight <int>
            --output <dir>

        - confirm: Manually reviews motion score PNGs.
            --input <*_motion.png>
            --threshold <int>
            --output <file.csv>
    """
    parser = argparse.ArgumentParser(description="Grade or confirm motion scores from medical images.")
    
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Subparser for grading
    parser_grade = subparsers.add_parser('grade', help='Automatically grade a set of AIM images')
    parser_grade.add_argument('--input', nargs='+', required=True, help='List of AIM image files (e.g., *.AIM)')
    parser_grade.add_argument('--stackheight', type=int, default=168, help='Stack height to evaluate')
    parser_grade.add_argument('--output', type=str, required=True, help='Directory to save graded results')

    # Subparser for confirmation
    parser_confirm = subparsers.add_parser('confirm', help='Manually confirm motion scores from PNG files')
    parser_confirm.add_argument('--input', nargs='+', required=True, help='List of motion PNGs (e.g., *motion.png)')
    parser_confirm.add_argument('--threshold', type=int, default=75, help='Confidence threshold (%) for automatic acceptance')
    parser_confirm.add_argument('--output', type=str, required=True, help='CSV file path to save grading results')

    return parser.parse_args()

def main():
    """
        Main entry point for the MotionScore CLI tool.

        Allows users to either:
        - Automatically grade a set of AIM images (`grade` mode).
        - Manually confirm motion scores from saved PNG outputs (`confirm` mode).

        The selected mode and arguments are parsed using argparse, and the appropriate
        function (`grade_images` or `confirm_images`) is called.

        Citation
        --------
        If you use this tool in your research or publication, please cite:

            Walle, M., Eggemann, D., Atkins, P.R., Kendall, J.J., Stock, K., Müller, R. and Collins, C.J., 2023.
            Motion grading of high-resolution quantitative computed tomography supported by deep convolutional neural networks.
            Bone, 166, p.116607.
            https://doi.org/10.1016/j.bone.2022.116607
    """
    print("=" * 80)
    print("If you use this tool, please cite:")
    print()
    print("   Walle, M., Eggemann, D., Atkins, P.R., Kendall, J.J., Stock, K., Müller, R. and Collins, C.J., 2023.")
    print("   Motion grading of high-resolution quantitative computed tomography supported")
    print("   by deep convolutional neural networks. Bone, 166, p.116607.")
    print("   https://doi.org/10.1016/j.bone.2022.116607")
    print("=" * 80)
    print()
    
    args = parse_args()

    if args.mode == 'grade':
        grade_images(args.input, args.stackheight, args.output)

    elif args.mode == 'confirm':
        confirm_images(args.input, args.threshold, args.output)

if __name__ == "__main__":
    main()