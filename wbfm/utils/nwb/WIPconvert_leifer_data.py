import os
import numpy as np
import scipy.io as sio
from scipy.optimize import curve_fit
import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter



def poly1(x, a, b):
    return a * x + b

def weighted_polyfit(x, y, w):
    """Weighted linear fit equivalent to MATLAB fit(...,'poly1','Weight',...)"""
    # Apply weights by scaling x,y
    W = np.sqrt(w)
    xw = x * W
    yw = y * W
    coeffs = np.polyfit(xw, yw, 1)
    return np.poly1d(coeffs)

def triple_flash_align(data_folder=None):
    """
    Python equivalent of tripleFlashAlign.m
    """

    if data_folder is None:
        raise ValueError("data_folder must be specified (MATLAB version uses GUI uipickfiles).")

    # Get data from yaml or avi metadata
    yaml_files = [f for f in os.listdir(data_folder) if f.endswith('.yaml')]
    if len(yaml_files) == 0:
        _, fluorAll, bfAll = AviFlashAlign(data_folder)
    else:
        _, fluorAll, bfAll = YamlFlashAlign(data_folder)

    # Load hiResData
    hi_res_path = os.path.join(data_folder, 'hiResData.mat')
    if os.path.exists(hi_res_path):
        hiResData = sio.loadmat(hi_res_path)['dataAll']
    else:
        hiResData = highResTimeTraceAnalysisTriangle4(data_folder)

    hiResFlashTime = hiResData['frameTime'][0, hiResData['flashLoc'][0]]
    bfFlashTime = bfAll['frameTime'][0, bfAll['flashLoc'][0]]
    fluorFlashTime = fluorAll['frameTime'][0, fluorAll['flashLoc'][0]]

    # Bandaid: if only 1 flash in each
    if (len(hiResFlashTime) == 1 and 
        len(bfFlashTime) == 1 and 
        len(fluorFlashTime) == 1):
        hiResFlashTime = np.array([hiResFlashTime, hiResFlashTime + 10])
        bfFlashTime = np.array([bfFlashTime, bfFlashTime + 10])
        fluorFlashTime = np.array([fluorFlashTime, fluorFlashTime + 10])

    # Pick reference with most flashes
    counts = [len(hiResFlashTime), len(bfFlashTime), len(fluorFlashTime)]
    most = np.argmax(counts)
    if most == 0:
        bestFlashTime = hiResFlashTime
    elif most == 1:
        bestFlashTime = bfFlashTime
    else:
        bestFlashTime = fluorFlashTime

    if (len(hiResFlashTime) == 0 or len(bfFlashTime) == 0 or len(fluorFlashTime) == 0):
        raise RuntimeError("Flashes not detected in all of the videos")

    # Align BF
    _, bf2fluor = flashTimeAlign2(bestFlashTime, bfFlashTime)
    flashDiff = bfFlashTime - bestFlashTime[bf2fluor]
    flashDiff -= np.min(flashDiff)
    f_bfTime = weighted_polyfit(bfFlashTime, bestFlashTime[bf2fluor], np.exp(-flashDiff**2))
    if f_bfTime.coeffs[0] < 0.1:
        f_bfTime = np.poly1d([1, 0])
    bfAll['frameTime'] = f_bfTime(bfAll['frameTime'])

    # Align Fluor
    _, bf2fluor = flashTimeAlign2(bestFlashTime, fluorFlashTime)
    flashDiff = fluorFlashTime - bestFlashTime[bf2fluor]
    flashDiff -= np.min(flashDiff)

    if len(fluorFlashTime) > 1:
        f_fluorTime = weighted_polyfit(fluorFlashTime, bestFlashTime[bf2fluor], np.exp(-flashDiff**2))
        if f_fluorTime.coeffs[0] < 0.1:
            f_fluorTime = np.poly1d([1, 0])
        fluorAll['frameTime'] = f_fluorTime(fluorAll['frameTime'])
    else:
        fluorAll['frameTime'] = fluorAll['frameTime'] - fluorFlashTime + bestFlashTime[bf2fluor]

    # Align hiRes
    _, bf2fluor = flashTimeAlign2(bestFlashTime, hiResFlashTime)
    flashDiff = hiResFlashTime - bestFlashTime[bf2fluor]
    flashDiff -= np.min(flashDiff)
    f_hiTime = weighted_polyfit(hiResFlashTime, bestFlashTime[bf2fluor], np.exp(-flashDiff**2))
    if f_hiTime.coeffs[0] < 0.1:
        f_hiTime = np.poly1d([1, 0])
    hiResData['frameTime'] = f_hiTime(hiResData['frameTime'])

    # Start time = first volume
    startTime = hiResData['frameTime'][hiResData['stackIdx'] == 1][0]
    hiResData['frameTime'] -= startTime
    fluorAll['frameTime'] -= startTime
    bfAll['frameTime'] -= startTime

    # Recover flash times
    bfAll['flashTime'] = bfAll['frameTime'][bfAll['flashLoc']]
    fluorAll['flashTime'] = fluorAll['frameTime'][fluorAll['flashLoc']]
    hiResData['flashTime'] = hiResData['frameTime'][hiResData['flashLoc']]

    return bfAll, fluorAll, hiResData


def smooth(x, window_len=200):
    """MATLAB-like smoothing with Savitzky-Golay filter"""
    return savgol_filter(x, window_length=window_len, polyorder=3, mode="interp")


def AviFlashAlign(data_folder):
    d = [f for f in os.listdir(data_folder) if f.startswith("LowMagBrain")]
    if len(d) > 1:
        raise RuntimeError("Multiple LowMagBrain folders found, need manual selection")
    elif len(d) == 1:
        avi_folder = os.path.join(data_folder, d[0])
    else:
        raise RuntimeError("AviFlashAlign:LowMagMissing - No LowMagBrain folder found")

    cam_files = [os.path.join(avi_folder, f) for f in os.listdir(avi_folder) if f.endswith(".avi")]
    if not cam_files:
        raise RuntimeError("AviFlashAlign:AVIMissing - No .avi files found")

    flash_files = [f.replace(".avi", "flashTrack.mat") for f in cam_files]

    if not flash_files or len(flash_files) < 2:
        raise RuntimeError("AviFlashAlign:FlashMissing - Missing flashTrack.mat files")

    # Load or compute flash tracks
    if os.path.exists(flash_files[0]):
        fluorFlash = loadmat(flash_files[0])["imFlash"].squeeze()
    else:
        fluorFlash = findFlash(cam_files[0])

    if os.path.exists(flash_files[1]):
        bfFlash = loadmat(flash_files[1])["imFlash"].squeeze()
    else:
        bfFlash = findFlash(cam_files[1])

    # Process traces
    bfFlash = bfFlash - smooth(bfFlash, 200)
    fluorFlash = fluorFlash - smooth(fluorFlash, 200)

    bfFlash -= np.min(bfFlash)
    fluorFlash -= np.min(fluorFlash)

    bfFlashloc = np.where(bfFlash > (np.mean(bfFlash) + 5*np.std(bfFlash)))[0]
    bfFlashloc = bfFlashloc[np.diff(np.insert(bfFlashloc, 0, -10)) >= 3]

    fluorFlashloc = np.where(fluorFlash > (np.mean(fluorFlash) + 5*np.std(fluorFlash)))[0]
    fluorFlashloc = fluorFlashloc[np.diff(np.insert(fluorFlashloc, 0, -10)) >= 3]

    # Load time from CamData.txt
    cam_data = pd.read_csv(os.path.join(avi_folder, "CamData.txt"), sep="\t")
    time = cam_data.iloc[:,1].values
    # Make times unique
    time = time + np.mean(np.diff(time))*0.001*np.arange(1, len(time)+1)

    bf2fluorIdx = np.arange(len(bfFlash))

    bfAll = {
        "frameTime": time,
        "flashTrack": bfFlash,
        "flashLoc": np.union1d(bfFlashloc, fluorFlashloc)
    }

    fluorAll = {
        "frameTime": time,
        "flashTrack": fluorFlash,
        "flashLoc": np.union1d(bfFlashloc, fluorFlashloc)
    }

    return bf2fluorIdx, fluorAll, bfAll


import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist

def YamlFlashAlign(data_folder):
    def classify_files(file_list):
        flashes = [f for f in file_list if "flash" in f.lower()]
        fluors = [f for f in file_list if "fluor" in f.lower()]
        yamls   = [f for f in file_list if "yaml"  in f.lower()]
        return flashes, fluors, yamls

    mat_files = [f for f in os.listdir(data_folder) if f.endswith(".mat")]
    flashes, fluors, yamls = classify_files(mat_files)

    # --- Fallback if not enough YAML or flash mat files ---
    if len(yamls) < 2 or len(flashes) < 2:
        yaml_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".yaml")]
        if not yaml_files:
            raise RuntimeError("Missing YAML mat files and raw .yaml not found. (MATLAB would ask user to select)")
        
        avi_files = [f for f in os.listdir(data_folder) if f.endswith(".avi") and "HUDS" not in f]
        avi_files = [os.path.join(data_folder, f) for f in avi_files]
        if not avi_files:
            raise RuntimeError("Missing flash mat files and no suitable .avi found. (MATLAB would ask user to select)")
        
        # Stub equivalents (MATLAB: findFlash, makeMatFileFromYaml)
        print("Would call findFlash(avi_files) and makeMatFileFromYaml(yaml_files) here")
        # TODO: implement these in Python

        # Re-scan mat files after making them
        mat_files = [f for f in os.listdir(data_folder) if f.endswith(".mat")]
        flashes, fluors, yamls = classify_files(mat_files)

    # --- Load data from mat files ---
    fluorFlash = bfFlash = fluorYaml = bfYaml = None

    for f in mat_files:
        path = os.path.join(data_folder, f)
        data = loadmat(path, squeeze_me=True, struct_as_record=False)

        if "flash" in f.lower() and "fluor" in f.lower():
            fluorFlash = data["imFlash"].squeeze()
            fluorFlash = fluorFlash - savgol_filter(fluorFlash, 200, 3)
        elif "flash" in f.lower():
            bfFlash = data["imFlash"].squeeze()
            bfFlash = bfFlash - savgol_filter(bfFlash, 200, 3)
        elif "yaml" in f.lower() and "fluor" in f.lower():
            fluorYaml = data["mcdf"].squeeze()
        elif "yaml" in f.lower():
            bfYaml = data["mcdf"].squeeze()

    if bfFlash is None or fluorFlash is None or bfYaml is None or fluorYaml is None:
        raise RuntimeError("Missing one of the required inputs (bfFlash, fluorFlash, bfYaml, fluorYaml)")

    # --- Frame times ---
    bfFrameTime = np.array([entry.TimeElapsed for entry in np.ravel(bfYaml)])
    bfFrameTime -= np.min(bfFrameTime)

    fluorFrameTime = np.array([entry.TimeElapsed for entry in np.ravel(fluorYaml)])
    fluorFrameTime -= np.min(fluorFrameTime)

    # --- Flash indices ---
    bfFlash -= np.min(bfFlash)
    fluorFlash -= np.min(fluorFlash)

    bfFlashloc = np.where(bfFlash > (np.mean(bfFlash) + 5 * np.std(bfFlash)))[0]
    fluorFlashloc = np.where(fluorFlash > (np.mean(fluorFlash) + 5 * np.std(fluorFlash)))[0]

    bfFlashTime = bfFrameTime[bfFlashloc]
    fluorFlashTime = fluorFrameTime[fluorFlashloc]

    # --- Align based on intervals ---
    if len(bfFlashTime) > 1 and len(fluorFlashTime) > 1:
        intervalDif = cdist(np.diff(bfFlashTime).reshape(-1, 1),
                            np.diff(fluorFlashTime).reshape(-1, 1))
        min_idx = np.unravel_index(np.argmin(intervalDif), intervalDif.shape)
        timeDif = bfFlashTime[min_idx[0]] - fluorFlashTime[min_idx[1]]
    else:
        timeDif = bfFlashTime[0] - fluorFlashTime[0]

    fluorFrameTime = fluorFrameTime + timeDif

    # Interpolate bf → fluor index mapping
    bf2fluorIdx = np.round(
        np.interp(bfFrameTime, fluorFrameTime, np.arange(len(fluorFrameTime)))
    ).astype(int)

    fluorAll = {
        "frameTime": fluorFrameTime,
        "flashLoc": fluorFlashloc,
        "flashTrack": fluorFlash
    }

    bfAll = {
        "frameTime": bfFrameTime,
        "flashLoc": bfFlashloc,
        "flashTrack": bfFlash
    }

    return bf2fluorIdx, fluorAll, bfAll



import numpy as np
from scipy.spatial.distance import cdist

def flashTimeAlign2(flashA, flashB):
    best = np.inf
    idxOut = None
    outputOffset = None

    for i in range(len(flashB)):
        for j in range(len(flashA)):
            offset = flashA[j] - flashB[i]
            ABdist = cdist(flashA.reshape(-1,1), (flashB+offset).reshape(-1,1))
            minDistAll = np.min(ABdist, axis=0)
            totalDist = np.sum(minDistAll)

            if totalDist < best:
                best = totalDist
                idxOut = [i, j]
                outputOffset = np.argmin(ABdist, axis=0)

    return idxOut, outputOffset


import os
import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter


def normalizeRange(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def highResTimeTraceAnalysisTriangle4(imFolder):
    camFrameData = pd.read_csv(os.path.join(imFolder, "CameraFrameData.txt"), sep="\t").values
    saveIdx = camFrameData[:,1]

    labJackData = pd.read_csv(os.path.join(imFolder, "LabJackData.txt"), sep="\t").values

    datFile = [f for f in os.listdir(imFolder) if f.startswith("sCMOS_Frames_U16_")][0]
    datFile = os.path.join(imFolder, datFile)

    row, col = getdatdimensions(datFile)
    datFlashRaw = findDatFlash(datFile, row, col, 10)

    datFlash = datFlashRaw - np.nanmean(datFlashRaw)
    datFlash = np.where(datFlash > np.nanstd(datFlashRaw)*10)[0]
    datFlash = datFlash[np.diff(np.insert(datFlash,0,-10)) >= 3]

    imageWave = normalizeRange(labJackData[:,3]) < 0.5
    zTrigger = labJackData[:,2]
    zTrigger2 = normalizeRange(zTrigger) > 0.5

    smoothKernal = np.median(np.diff(np.where(np.diff(zTrigger2)==1)))//2
    zTrigger = savgol_filter(zTrigger, int(smoothKernal)|1, 3)
    zWave = labJackData[:,1]

    daqSaveFrame = labJackData[:,4]
    saveSpikes = np.diff(daqSaveFrame) > 0

    stackIdx = np.cumsum(np.abs(np.diff(zTrigger/np.std(zTrigger))) > .1)
    stackIdx = np.insert(stackIdx, 0, 0)
    stackIdx = stackIdx[saveSpikes]

    timeAll = np.where(saveSpikes)[0]/1000.0

    imageIdx = camFrameData[:,0]
    imSTD = camFrameData[:,-1]
    imSTD = imSTD[np.diff(saveIdx) > 0]
    imSTD = imSTD - np.min(imSTD)

    imageIdx = imageIdx[np.diff(saveIdx) > 0]
    imageIdx = imageIdx - np.min(imageIdx) + 1

    dataAll = {
        "Z": zWave[saveSpikes],
        "flashLoc": datFlash,
        "imageIdx": imageIdx,
        "frameTime": timeAll,
        "stackIdx": stackIdx,
        "imSTD": imSTD,
        "xPos": labJackData[:,6][saveSpikes] if labJackData.shape[1]>7 else [],
        "yPos": labJackData[:,7][saveSpikes] if labJackData.shape[1]>7 else []
    }

    # Save for compatibility
    np.savez(os.path.join(imFolder, "hiResData.npz"), **dataAll)
    with open(os.path.join(imFolder, "submissionParameters.txt"), "w") as f:
        f.write(f"NFrames {np.max(stackIdx)}")

    return dataAll


import os
import numpy as np
import tifffile
import cv2
from scipy.io import savemat

def findFlash(imFolderIn, custom_roi=None):
    """
    imFolderIn : str
        Folder containing .tif images or path to .avi file.
    custom_roi : np.ndarray[bool], optional
        Boolean mask (same shape as one frame). If None, use whole frame.
    """

    if os.path.isdir(imFolderIn):
        # TIFF stack
        imFiles = [os.path.join(imFolderIn, f) for f in os.listdir(imFolderIn) if f.endswith(".tif")]
        imFiles.sort()
        stackSize = len(imFiles)
        initialIm = tifffile.imread(imFiles[0])
        aviFlag = False
    elif imFolderIn.endswith(".avi"):
        cap = cv2.VideoCapture(imFolderIn)
        stackSize = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        initialIm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ret else None
        aviFlag = True
    else:
        raise ValueError("Input must be a TIFF folder or an AVI file")

    if custom_roi is None:
        roiFlash = np.ones_like(initialIm, dtype=bool)
    else:
        roiFlash = custom_roi

    imFlash = np.zeros(stackSize, dtype=float)

    if aviFlag:
        cap = cv2.VideoCapture(imFolderIn)
        for i in range(stackSize):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imFlash[i] = np.mean(gray[roiFlash])
        cap.release()
    else:
        for i, fname in enumerate(imFiles):
            frame = tifffile.imread(fname)
            imFlash[i] = np.mean(frame[roiFlash])

    # Save alongside
    if aviFlag:
        save_name = imFolderIn.replace(".avi", "flashTrack.mat")
    else:
        save_name = os.path.join(imFolderIn, "flashTrack.mat")

    savemat(save_name, {"imFlash": imFlash})

    return imFlash


import numpy as np
import os

def findDatFlash(datFile, rows=None, cols=None, rowSearch=None):
    """
    datFile : str
        Path to .dat file with uint16 frames.
    rows, cols : int
        Image dimensions. If None, try parsing from filename.
    rowSearch : int
        Number of rows to average over. Defaults to rows.
    """

    if not datFile.endswith(".dat"):
        raise ValueError("File must be a .dat binary file")

    if rows is None or cols is None:
        rows, cols = getdatdimensions(datFile)

    if rowSearch is None:
        rowSearch = rows

    bytes_per_frame = rows * cols * 2  # uint16 = 2 bytes
    filesize = os.path.getsize(datFile)
    stackSize = filesize // bytes_per_frame

    imFlash = np.zeros(stackSize, dtype=float)

    with open(datFile, "rb") as f:
        for i in range(stackSize):
            # Read only first rowSearch rows
            nvals = rows * rowSearch
            data = np.fromfile(f, dtype=np.uint16, count=nvals)
            if data.size < nvals:
                break
            imFlash[i] = np.mean(data)
            # Skip rest of frame
            f.seek(bytes_per_frame - nvals*2, os.SEEK_CUR)

    return imFlash


def getdatdimensions(filename):
    """
    Parse image dimensions from .dat filename, e.g.
    sCMOS_Frames_U16_1024x512.dat
    """
    base = os.path.basename(filename)
    try:
        sizepart = base.split("_")[-1].replace(".dat", "")
        rows, cols = sizepart.split("x")
        return int(rows), int(cols)
    except Exception:
        raise ValueError(f"Could not parse dimensions from {filename}")
