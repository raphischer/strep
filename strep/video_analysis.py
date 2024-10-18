import os
import time
import warnings

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
from tqdm import tqdm


# Function to get the total frame count in the video
def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

# Function to a single video and save frames with high red/green levels
def extract_frames(video_path, frame_filename, state=None, resize=(100, 50), exp_experiments=1000):
    zf = len(str(exp_experiments*2))
    total_frames = get_frame_count(video_path)
    cap = cv2.VideoCapture(video_path)
    fr_cnt, colors = 0, np.zeros((total_frames, 3)) # store RGB colors

    # Process video frames with tqdm progress bar
    with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: # end of video
                break
            
            # opencv resize are super fast, so resize before taking means
            frame_small = cv2.resize(frame, resize)
            for chan in range(frame_small.shape[2]): # rgb
                colors[fr_cnt,chan] = np.mean(frame_small[:,:,chan])

            # check for state, store frame if switch detected
            if state is None: # init state
                state = 1
            elif state % 2 == 0:
                if not colors[fr_cnt,1] > colors[fr_cnt,0] or not colors[fr_cnt,1] > colors[fr_cnt,2]: # end of profiling
                    cv2.imwrite(frame_filename.format(fc=str(state).zfill(zf)), frame)
                    state += 1
            else:
                if colors[fr_cnt,1] > colors[fr_cnt,0] and colors[fr_cnt,1] > colors[fr_cnt,2]: # start of profiling
                    cv2.imwrite(frame_filename.format(fc=str(state).zfill(zf)), frame)
                    state += 1

            # update progressbar
            fr_cnt += 1
            pbar.update(1)

    cap.release()
    return state, colors

def process_videos(video_directory, output_folder):
    videos = sorted(os.listdir(video_directory))
    frame_name = os.path.join(output_folder, "frame_{fc}.jpg")
    state = None # None: not init    0: profiling    1: preparing next
    for v_idx, vname in enumerate(videos):
        print(f'\n\nPROCESSING VIDEO {v_idx+1:<2} / {len(videos)}\n\n')
        vfile = os.path.join(video_directory, vname)
        state, colors = extract_frames(vfile, frame_name, state=state)
        np.save(os.path.join(output_folder, f"colors_{v_idx}.npy"), colors)
        # fig = go.Figure([
        #     go.Scatter(x=np.arange(total_frames), y=colors[:,0], name='B', line={'color': 'blue'}),
        #     go.Scatter(x=np.arange(total_frames), y=colors[:,1], name='G', line={'color': 'green'}),
        #     go.Scatter(x=np.arange(total_frames), y=colors[:,2], name='R', line={'color': 'red'})
        # ])
        # fig.show()

# Mouse callback function to draw a rectangle and save the ROI automatically
def draw_rectangle(event, x, y, flags, param):
    global roi, drawing, start_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)  # Store the starting point of the rectangle

    elif event == cv2.EVENT_MOUSEMOVE:
        try:
            if drawing:
                # Temporary rectangle as the user drags the mouse
                frame_copy = param.copy()
                cv2.rectangle(frame_copy, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("Select ROI", frame_copy)
        except NameError:
            pass

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (start_point[0], start_point[1], x, y)  # Store ROI as (x1, y1, x2, y2)
        cv2.rectangle(param, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", param)
        print(f"ROI saved: Top-left: {start_point}, Bottom-right: ({x}, {y})")
        cv2.destroyWindow("Select ROI")  # Automatically close the window when done

def select_roi(frame):
    # Display the frame and set the mouse callback
    cv2.imshow("Select ROI", frame)
    cv2.setMouseCallback("Select ROI", draw_rectangle, frame)
    print("Draw a rectangle to select the region of interest (ROI).")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # make sure that the coordinates are in correct ordner, no matter how the user draws the rectangle
    x1, y1, x2, y2 = roi
    if x1 > x2:
        fr, x1 = x1, x2
        x2 = fr
    if y1 > y2:
        fr, y1 = y1, y2
        y2 = fr
    return x1, y1, x2, y2
    
def detect_ocr(single_frame, ocr_func, preprocessor):
    # TODO speed up by merging all images, preprocessing them all together, and doing a row-wise ocr detection
    if callable(preprocessor) and preprocessor.__name__ == "<lambda>":
        frame_thresh = preprocessor(single_frame)
    else: # holds the fours parameters
        block_size, c_value, kernel_size, erosion_iterations = preprocessor[0], preprocessor[1], preprocessor[2], preprocessor[3]
        if not isinstance(block_size, int):
            block_size = np.round(block_size).astype(int)
        if not isinstance(kernel_size, int):
            kernel_size = np.round(kernel_size).astype(int)
        if not isinstance(erosion_iterations, int):
            erosion_iterations = np.round(erosion_iterations).astype(int)
        frame_thresh = apply_preprocessing(single_frame, block_size, c_value, kernel_size, erosion_iterations)
    ocr = ocr_func( cv2.cvtColor(frame_thresh, cv2.COLOR_GRAY2RGB) )
    return ocr, frame_thresh

def apply_preprocessing(image, block_size, c_value, kernel_size, erosion_iterations):
    # Ensure block_size is odd and greater than 1
    if block_size % 2 == 0:
        block_size += 1
    resized = cv2.resize(image, (0, 0), fx=0.5, fy=0.5) # resizing makes everything faster -> TODO move outside of this function
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) # convert to gray
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_value) # threshold
    kernel = np.ones((kernel_size, kernel_size), np.uint8) # create erosion & dilation kernel
    dilated_image = cv2.dilate(thresh, kernel, iterations=erosion_iterations)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=erosion_iterations)
    return eroded_image

def display_with_text(images, texts, text_box_width=400):
    stacked = []
    for image, text in zip(images, texts):        
        # Write text on white canvas
        canvas = np.ones((image.shape[0], text_box_width, 3), dtype=np.uint8) * 255
        font, font_scale, thickness, line_height, y0 = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 25, 30
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * line_height
            cv2.putText(canvas, line, (10, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        # Concatenate the image with the text box horizontally
        stacked.append( np.hstack((image, canvas)) )
    # Display the vertically stacked images with text areas
    cv2.imshow('Preprocessing and OCR', np.vstack(stacked))

def update_preprocessing(x, get_ocr):
    try: # Get current positions of trackbars
        block_size = cv2.getTrackbarPos('Block Size', 'Preprocessing and OCR')
        c_value = cv2.getTrackbarPos('C Value', 'Preprocessing and OCR')
        kernel_size = cv2.getTrackbarPos('Kernel Size', 'Preprocessing and OCR')
        erosion_iterations = cv2.getTrackbarPos('Erosion Iterations', 'Preprocessing and OCR')
    except Exception:
        return # only happen during initialization of window

    ocr, imgs = [], []
    for frame in frames:
        # Apply preprocessing with current parameters and store the OCR results
        processed_image = apply_preprocessing(frame, block_size, c_value, kernel_size, erosion_iterations)
        ocr.append( get_ocr(processed_image) )
        imgs.append( cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR) )

    # Display the processed image with the OCR text side by side
    display_with_text(imgs, ocr)

# Function to create an interactive window for preprocessing controls and OCR output
def interactive_preprocessing_with_ocr(images, get_ocr):
    global frames
    frames = images.copy()  # Store the image globally for access inside trackbar callback

    # Create window and trackbars for adjusting preprocessing parameters
    cv2.namedWindow('Preprocessing and OCR')
    cv2.createTrackbar('Block Size', 'Preprocessing and OCR', 21, 50, lambda x: update_preprocessing(x, get_ocr))
    cv2.createTrackbar('C Value', 'Preprocessing and OCR', 10, 20, lambda x: update_preprocessing(x, get_ocr))
    cv2.createTrackbar('Kernel Size', 'Preprocessing and OCR', 1, 20, lambda x: update_preprocessing(x, get_ocr))
    cv2.createTrackbar('Erosion Iterations', 'Preprocessing and OCR', 1, 10, lambda x: update_preprocessing(x, get_ocr))
    update_preprocessing(0, get_ocr) # call once for initial display

    # Keep the window open until the user presses 'Esc'
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # Escape key to exit
            break

    block_size = cv2.getTrackbarPos('Block Size', 'Preprocessing and OCR')
    c_value = cv2.getTrackbarPos('C Value', 'Preprocessing and OCR')
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'Preprocessing and OCR')
    erosion_iterations = cv2.getTrackbarPos('Erosion Iterations', 'Preprocessing and OCR')
    cv2.destroyAllWindows()
    return lambda im: apply_preprocessing(im, block_size, c_value, kernel_size, erosion_iterations)

def get_manual_ocr(image, frame_name, next_known, width=100, height=12):
    # Crop the image to remove all-white rows/columns
    rows, cols = np.any(image == 0, axis=1), np.any(image == 0, axis=0)
    cropped_image = image[np.ix_(rows, cols)]    

    # Rescale the cropped image to fixed size for command line output
    resized_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Display binary image as pixel art in the terminal
    print("-----------------------------------------------------")
    for y in range(height):
        print(''.join(list(map(lambda v: '█' if v == 0 else ' ', resized_image[y,:]))))

    # Prompt the user for manual OCR correction in the terminal
    corrected_text = input(f"\nPlease type the displayed float number and hit enter ({next_known}, current frame is {frame_name}): ")
    return corrected_text

def param_evaluate(roi_frames, params):
    global DRF_T0, DRF_TMAX, DRF_BEST_X, DRF_BEST_F
    # stop criteria
    elapsed = time.time() - DRF_T0
    if elapsed > DRF_TMAX:
        raise Exception
    # collect  results
    ocr_results = []
    for idx, fr in enumerate(roi_frames):
        try:
            ocr_results.append( detect_ocr(fr, ocr_func, params)[0] )
            assert len(ocr_results[-1]) > 0
        except Exception:
            print(f'  failed to evaluate    {str(params):<55} - OCR crashed for frame {idx:<3} - {DRF_TMAX-elapsed:5.3f}s remaining', end='\r')
            return np.inf
    # calculate metrics
    lengths, floats = [], []
    for ocr in ocr_results:
        lengths.append(len(ocr))
        try:
            floats.append(float(ocr))
        except Exception:
            continue
    dev_length = np.std(lengths)
    successful_floats = 1 - len(floats) / len(lengths)
    min_neg_diff = max(np.min(pd.Series(floats).diff()) * -1, 0)
    if min_neg_diff > 0:
        min_neg_diff /= min_neg_diff # 0 if strictly ordered, 1 if there is an ordering mistake in the results
    func = dev_length + successful_floats + min_neg_diff
    print(f'  success with evaluate {str(params):<55} - f(x): {dev_length:4.2f} + {successful_floats:4.2f} + {min_neg_diff:5.2f} - {DRF_TMAX-elapsed:5.3f}s remaining\n')
    if func < DRF_BEST_F:
        DRF_BEST_F = func
        DRF_BEST_X = params
    return func

# Example usage
video_directory = r'D:/Videos/24 10 energy/imagenet_1_2024-10-17_10-13-03'
output_folder = os.path.basename(video_directory)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# filter for important video frames with very high or low green amounts (start / end of profiling)
# process_videos(video_directory, output_folder)

# init OCR with tesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
os.environ['TESSDATA_PREFIX'] = r'D:\Repos\Tesseract_sevenSegmentsLetsGoDigital\Trained data'
ocr_func = lambda im: pytesseract.image_to_string(im, lang='lets', config='--psm 6').replace('\n', '').replace(',', '.').replace(' ', '').replace('-', '') # get lets from https://github.com/adrianlazaro8/Tesseract_sevenSegmentsLetsGoDigital/tree/master

# identify preprocessing hyperparameters
# TODO handle with argparse and local config file
DRF_TMAX = 5 * 60
roi = (1662, 601, 1843, 665)
bounds = [(1, 50), (1, 20), (3, 6), (1, 5)]
# roi = select_roi(cv2.imread(os.path.join(output_folder, frame_names[0])))
x1, y1, x2, y2 = roi
frame_names = [fname for fname in sorted(os.listdir(output_folder)) if 'frame_' in fname]
roi_frames = [cv2.imread(os.path.join(output_folder, fname))[y1:y2, x1:x2] for fname in frame_names]
# DRF_BEST_X = np.array([21, 10, 3, 1]) # block_size, c_value, kernel_size, erosion_iterations
DRF_BEST_X, DRF_BEST_F, DRF_T0 = None, np.inf, time.time()
warnings.filterwarnings("ignore")
try:
    dual_annealing(lambda x: param_evaluate(roi_frames, x), bounds=bounds, no_local_search=True) # minimizer_kwargs={'method': 'Nelder-Mead', 'bounds': bounds, 'options': {'maxfun': 2}})
except Exception:
    print('stopped due to time limit')
# or control interactively
# test_frames = [cv2.imread(os.path.join(output_folder, fr))[y1:y2, x1:x2] for fr in np.random.choice(frame_names, size=7)]
# preprocessor = interactive_preprocessing_with_ocr(test_frames, ocr_func)

# run OCR for all frames
ocr_out = {}
for idx, frame_name in tqdm(enumerate(frame_names)):
    prev_name = frame_names[idx-1]
    error, val = False, np.nan
    ocr, frame = detect_ocr(cv2.imread(os.path.join(output_folder, frame_name))[y1:y2, x1:x2], ocr_func, DRF_BEST_X)
    try:
        assert len(ocr) == 5
        val = float(ocr)
        if idx > 0 and isinstance(ocr_out[prev_name]['value'], float):
            assert ocr_out[prev_name]['value'] <= val
    except Exception:
        last_known = ocr_out[prev_name]['value'] if idx > 0 else 0
        while not error:
            ocr = get_manual_ocr(frame, frame_name, f'last number was {last_known}')
            try:
                val = float(ocr)
                error = True
            except Exception:
                print(f'Incorrect input "{ocr}"!')
    ocr_out[frame_name] = {'ocr': ocr, 'value': val, 'manual': error}
    ocr_fname = frame_name.replace('frame', 'ocr').replace('.jpg', f'_{ocr.replace(".", "-")}.jpg')
    cv2.imwrite(os.path.join(output_folder, ocr_fname), frame)

# traverse backwards to find any new errors relating to manual correction
print('Now traversing backwards to find additional errors')
for idx, frame_name in tqdm(enumerate(reversed(frame_names))):
    if idx == len(ocr_out) - 1 or idx == 0:
        continue
    prev_name, next_name = frame_names[len(ocr_out)-idx-2], frame_names[len(ocr_out)-idx]
    last, current, next = ocr_out[prev_name], ocr_out[frame_name], ocr_out[next_name]
    if current['value'] < last['value'] or current['value'] > next['value']:
        error = False
        ocr, frame = detect_ocr(cv2.imread(os.path.join(output_folder, frame_name))[y1:y2, x1:x2], ocr_func, DRF_BEST_X)
        while not error:
            manual_input = get_manual_ocr(frame, frame_name, f'previous is {last["value"]}, next is {next["value"]}')
            try:
                ocr_out[frame_name]['value'] = float(manual_input)
            except Exception:
                print(f'Incorrect input "{manual_input}"!')
            try:
                assert ocr_out[frame_name]['value'] < next['value']
                ocr_out[frame_name]['manual'] = True
                error = True
                # delete and re-write already written ocr file
                ocr_fname = frame_name.replace('frame', 'ocr').replace('.jpg', f'_{ocr.replace(".", "-")}.jpg')
                os.remove(os.path.join(output_folder, ocr_fname))
                ocr_fname = frame_name.replace('frame', 'ocr').replace('.jpg', f'_{manual_input.replace(".", "-")}.jpg')
                cv2.imwrite(os.path.join(output_folder, ocr_fname), frame)
            except Exception:
                print(f'Incorrect input - input number ({manual_input}) cannot be bigger than the following value ({next["value"]})!')

# write the final summary
df = pd.DataFrame(ocr_out).transpose()
df['val_diff'] = df["value"].diff()
df['still_errors'] = df['val_diff'] < 0
df.to_csv(os.path.join(output_folder, 'video_data.csv'))
if not df["value"].is_monotonic_increasing:
    print('Still encountered errors in the following rows and frames:\n')
    print(df[df['val_diff'] < 0].index)
