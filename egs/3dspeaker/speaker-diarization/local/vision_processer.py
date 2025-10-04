# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script uses pretrained models to perform speaker visual embeddings extracting.
This script use following open source models:
    1. Face detection: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
    2. Active speaker detection: TalkNet, https://github.com/TaoRuijie/TalkNet-ASD
    3. Face quality assessment: https://modelscope.cn/models/iic/cv_manual_face-quality-assessment_fqa
    4. Face recognition: https://modelscope.cn/models/iic/cv_ir101_facerecognition_cfglint
Processing pipeline: 
    1. Face detection (input: video frames)
    2. Active speaker detection (input: consecutive face frames, audio)
    3. Face quality assessment (input: video frames)
    4. Face recognition (input: video frames)
"""


import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
import os, time, torch, cv2, pickle, python_speech_features
from facenet_pytorch import MTCNN

import vision_tools.active_speaker_detection as active_speaker_detection
import vision_tools.face_recognition as face_recognition
import vision_tools.face_quality_assessment as face_quality_assessment


class VisionProcesser():
    def __init__(
        self, 
        video_file_path, 
        audio_file_path, 
        audio_vad, 
        out_feat_path, 
        onnx_dir, 
        conf, 
        device='cpu', 
        device_id=0, 
        out_video_path=None
        ):
        # read audio data and check the samplerate.
        self.fs, self.audio = wavfile.read(audio_file_path)
        assert self.fs == 16000, '[ERROR]: Samplerate of wav must be 16000'
        # convert time interval to integer sampling point interval.
        audio_vad = [[int(i*16000), int(j*16000)] for (i, j) in audio_vad]
        self.video_id = os.path.basename(video_file_path).rsplit('.', 1)[0]

        # read video data
        self.cap = cv2.VideoCapture(video_file_path)
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        assert self.fps==25, '[ERROR]: The fps of input video must be 25.'
        self.asvf_ratio = 16000 / 25  # audio samples per video frame
        print('video %s info: w: {}, h: {}, count: {}, fps: {}'.format(w, h, self.count, self.fps) % self.video_id)

        # initial vision models
        self.face_detector = MTCNN( # explain for the parameters: https://blog.csdn.net/m0_49963403/article/details/136160453
          image_size=160, margin=0, min_face_size=20,
          thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
          device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
          keep_all=True
        )
        self.speaker_detector = active_speaker_detection.ASDTalknet(onnx_dir, device, device_id)
        self.face_quality_evaluator = face_quality_assessment.FaceQualityAssess(onnx_dir, device, device_id)
        self.face_embs_extractor = face_recognition.FaceRecIR101(onnx_dir, device, device_id)

        # store facial feats along with the necessary information.
        self.active_facial_embs = {'frameI':np.empty((0,), dtype=int), 'feat':np.empty((0, 512), dtype=np.float32)}

        self.audio_vad = audio_vad
        self.out_video_path = out_video_path
        self.out_feat_path = out_feat_path

        self.min_track = conf['min_track']  # face track中第一张、最后一张人脸在原始视频中的最少相隔帧数
        self.num_failed_det = conf['num_failed_det']  # face track中相邻两张人脸在原始视频中的最大相隔帧数
        self.crop_scale = conf['crop_scale']  # 在为 talknet准备数据时，在人脸检测框的基础上，适当放大，以提供更丰富的上下文
        self.min_face_size = conf['min_face_size']  # face track中 max(face平均宽度, face平均高度) 的最小值
        self.face_det_stride = conf['face_det_stride']  # 每隔多少帧进行一次人脸检测。虽然后续face tracking会插值补全，以便于ASD，但是最后一步的 embedding 提取仍然只对检测（而非插值）得到的人脸进行。
        self.shot_stride = conf['shot_stride']  # treat every 'shot_stride' frames as a processing unit for face tracking and subsequent steps

        if self.out_video_path is not None:
            # save the active face detection results video (for debugging).
            self.v_out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (int(w), int(h)))

        # record the time spent by each module.
        self.elapsed_time = {'faceTime':[], 'trackTime':[], 'cropTime':[],'asdTime':[], 'visTime':[], 'featTime':[]}

    def run(self):
        frames, face_det_frames = [], []
        # process each video segment
        for [audio_sample_st, audio_sample_ed] in self.audio_vad: # start/end sample index in corresponding audio
            # frame_st and frame_ed are the starting and ending frames of current segment.
            frame_st, frame_ed = int(audio_sample_st/self.asvf_ratio), int(audio_sample_ed/self.asvf_ratio)
            num_frames = frame_ed - frame_st + 1
            # go to frame 'frame_st'.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_st)
            index = 0
            for _ in range(num_frames):
                ## read current frame
                ret, frame = self.cap.read()  # ret(bool): read success or not; frame(ndarray): (h, w, 3), bgr format
                if not ret:
                    break
                ## record frames
                if index % self.face_det_stride==0:
                    face_det_frames.append(frame) # frames for face detection
                frames.append(frame)  # all frames in current shot
                ## process each shot
                if (index + 1) % self.shot_stride==0:
                    ### get corresponding audio of current shot
                    audio = self.audio[int((frame_st + index + 1 - self.shot_stride)*self.asvf_ratio):int((frame_st + index + 1)*self.asvf_ratio)]
                    ### process
                    self.process_one_shot(frames, face_det_frames, audio, frame_st + index + 1 - self.shot_stride)
                    ### reset
                    frames, face_det_frames = [], []
                index += 1
            if len(frames) != 0:  # process the remaining frames as a shot
                audio = self.audio[int((frame_st + index - len(frames))*self.asvf_ratio):int((frame_st + index)*self.asvf_ratio)]
                self.process_one_shot(frames, face_det_frames, audio, frame_st + index - len(frames))
                frames, face_det_frames = [], []

        self.cap.release()
        if self.out_video_path is not None:
            self.v_out.release()

        # save results: For each detected frame with one active speaker(with high quality face), record its timepoint and facial embedding
        active_facial_embs = {'embeddings':self.active_facial_embs['feat'], 'times': self.active_facial_embs['frameI']*0.04}
        pickle.dump(active_facial_embs, open(self.out_feat_path, 'wb'))

        # print elapsed time
        all_elapsed_time = 0
        for k in self.elapsed_time:
            all_elapsed_time += sum(self.elapsed_time[k])
            self.elapsed_time[k] = sum(self.elapsed_time[k])
        elapsed_time_msg = 'The total processing time for %s is %.2fs, including' % (self.video_id, all_elapsed_time)
        for k in self.elapsed_time:
            elapsed_time_msg += ' %s %.2fs,'%(k, self.elapsed_time[k])
        print(elapsed_time_msg[:-1]+'.')

    def process_one_shot(self, frames, face_det_frames, audio, frame_st=None):
        # Detect faces in each frame of face_det_frames. 
        # Return dets, a list of length len(face_det_frames), each element is a list of dict for each detected face in that frame
        curTime = time.time()
        dets = self.face_detection(face_det_frames)
        faceTime = time.time()

        # Get all face tracks of detected faces
        allTracks, vidTracks = [], []
        allTracks.extend(self.track_shot(dets))
        trackTime = time.time()

        # Get cropped face frames and audio segments for each track.
        ## vidTracks is a list of length len(allTracks), each element is a dict containing the cropped face frames and audio segment of the track
        for ii, track in enumerate(allTracks):
            vidTracks.append(self.crop_video(track, frames, audio))
        cropTime = time.time()

        # Calculate active speaker scores for each track
        ## scores is a list of length len(allTracks), each element is a numpy array of shape (num_frames, ), representing the active speaker scores for each frame in the track
        scores = self.evaluate_asd(vidTracks)
        asdTime = time.time()

        # Extract facial embeddings for detected frames with one active speaker(with high quality face)
        active_facial_embs = self.evaluate_fr(frames, vidTracks, scores)
        ## merge active_facial_embs of current shot to self.active_facial_embs
        self.active_facial_embs['frameI'] = np.append(self.active_facial_embs['frameI'], active_facial_embs['frameI'] + frame_st)
        self.active_facial_embs['feat'] = np.append(self.active_facial_embs['feat'], active_facial_embs['feat'], axis=0)
        featTime = time.time()

        if self.out_video_path is not None:
            self.visualization(frames, vidTracks, scores)
        visTime = time.time()

        # record elapsed time of each module
        self.elapsed_time['faceTime'].append(faceTime-curTime)
        self.elapsed_time['trackTime'].append(trackTime-faceTime)
        self.elapsed_time['cropTime'].append(cropTime-trackTime)
        self.elapsed_time['asdTime'].append(asdTime-cropTime)
        self.elapsed_time['featTime'].append(featTime-asdTime)
        if self.out_video_path is not None:
            self.elapsed_time['visTime'].append(visTime-featTime)

    def face_detection(self, frames):
        """
        Perform face detection on a sequence of video frames.

        Args:
          frames (list of numpy.ndarray): 
            A list of video frames, where each frame is represented as a 
            NumPy array of shape (H, W, C) in BGR format. 

        Returns:
          list of list of dict: 
            A nested list where each inner list corresponds to the detections 
            for a single frame. Each detection is represented as a dictionary 
            with the following keys:
            - 'frame' (int): The frame index in the video sequence, adjusted 
              by the face detection stride.
            - 'bbox' (list of float): The bounding box coordinates of the 
              detected face in the format [x1, y1, x2, y2].
            - 'conf' (float): The confidence score of the detection, 
              indicating the likelihood of the bounding box containing a face.
        """
        def box_filter(boxes, probs, min_size, min_prob):
            if boxes is None or probs is None:
                return torch.tensor([]), torch.tensor([]), torch.tensor([])
            num = boxes.shape[0]
            true_boxes = np.maximum(boxes, 0)
            filtered_index = [i for i in range(num) if min(true_boxes[i][3]-true_boxes[i][1], true_boxes[i][2]-true_boxes[i][0]) >= min_size and probs[i] >= min_prob]
            if len(filtered_index) == 0:
                return torch.tensor([]), torch.tensor([]), torch.tensor([])
            else:
                true_boxes = torch.from_numpy(true_boxes[filtered_index])
                true_box_probs = torch.from_numpy(probs[filtered_index])
                filtered_index = torch.tensor(filtered_index)
                return true_boxes, true_box_probs, filtered_index
        
        dets = []
        for fidx, image in enumerate(frames): # process each frame
            image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to rgb, ndarray (h, w, 3)
            bboxes, probs = self.face_detector.detect(image_input)  # boxes(np.array) of shape (n, 4), probs(np.array) of shape (n,)
            bboxes, probs, _ = box_filter(bboxes, probs, min_size=30, min_prob=0.8)
            bboxes = torch.cat([bboxes, probs.reshape(-1, 1)], dim=-1)  # (n_faces, 5), (x1, y1, x2, y2, conf)
            dets.append([])
            for bbox in bboxes:
                frame_idex = fidx * self.face_det_stride
                dets[-1].append({'frame':frame_idex, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
        return dets

    def bb_intersection_over_union(self, boxA, boxB, evalCol=False):
        # IOU Function to calculate overlap between two image
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if evalCol == True:
            iou = interArea / float(boxAArea)
        else:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def track_shot(self, scene_faces):
        """
        Face tracking: Get as many face tracks as possible in a video shot, accoding to the IOU of detected faces and num_failed_det.

        Args:
          scene_faces (list of list of dict): A list where each element represents a frame, and each frame contains a list of detected faces. 
            Each face is represented as a dictionary with the following keys:
              - 'frame' (int): The frame index where the face was detected.
              - 'bbox' (list of float): The bounding box of the face in the format [x_min, y_min, x_max, y_max].

        Returns:
          list of dict: A list of face tracks. Each track is represented as a dictionary with the following keys:
            - 'frame' (numpy.ndarray): A 1D array of continuous frame indices corresponding to the track.
            - 'bbox' (numpy.ndarray): A 2D array of interpolated bounding boxes for each frame in the track. 
              The shape is (num_frames, 4), where each row represents a bounding box in the format [x_min, y_min, x_max, y_max].

        Notes:
          - A face track is formed by linking consecutive detected faces based on IOU and temporal continuity.
          - Missing bounding boxes in a track are filled using linear interpolation.
          - Tracks are filtered to ensure they meet the minimum length (`self.min_track`) and face size (`self.min_face_size`) criteria.
          - The `self.num_failed_det` parameter determines the maximum number of frames a face can be undetected while still being part of the same track.
          - By interpolation, len(frame) = len(bbox) = number of frames between the first and last detected face in original video.
        """
        # Face tracking, only take frame index and box info of detected faces into account.
        tracks = []
        while True:   # continuously search for consecutive faces.
            # Try to find a new face track
            track = []
            for frame_faces in scene_faces: # 依次处理每一帧
                for face in frame_faces:
                    ## 初始化：将当前帧的不在track中的第一个人脸加入track
                    if track == []:
                        track.append(face)
                        frame_faces.remove(face)
                        break
                    ## 如果num_failed_det帧内，存在其它人脸，且与track中最后一张人脸的IOU>0.5，则将该人脸加入track
                    elif face['frame'] - track[-1]['frame'] <= self.num_failed_det:  # the face does not interrupt for 'num_failed_det' frame.
                        iou = self.bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                        # minimum IOU between consecutive face.
                        if iou > 0.5:
                            track.append(face)
                            frame_faces.remove(face)
                            break
                    else:
                        break
            if track == []:
                break
            # if the new track is valid, interpolate missing boxes and add to tracks
            elif len(track) > 1 and track[-1]['frame'] - track[0]['frame'] + 1 >= self.min_track:
                ## get frame_idx(in original scale, 0 means the first frame of current shot) and bbox info of each face in the track
                frame_num = np.array([ f['frame'] for f in track ])
                bboxes = np.array([np.array(f['bbox']) for f in track])
                ## get bbox info of each frame by linear interpolation
                frameI = np.arange(frame_num[0], frame_num[-1]+1)
                bboxesI = []
                for ij in range(0, 4):
                    interpfn  = interp1d(frame_num, bboxes[:,ij]) # missing boxes can be filled by interpolation.
                    bboxesI.append(interpfn(frameI))
                bboxesI  = np.stack(bboxesI, axis=1)
                ## filter out low-quality tracks
                if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > self.min_face_size:  # need face size > min_face_size
                    tracks.append({'frame':frameI,'bbox':bboxesI})
        return tracks

    def crop_video(self, track, frames, audio):
        """
        Get cropped video of a face track and the corresponding audio segment to prepare for active speaker detection.

        Args:
          track (dict): current face track in the form of dictionary with the following keys:
            - 'frame' (numpy.ndarray): A 1D array of continuous frame indices corresponding to the track.
            - 'bbox' (numpy.ndarray): A 2D array of shape (num_frames, 4), containing the interpolated bounding boxes for each frame in the track.
          
          frames (list of numpy.ndarray): A list of original video frames (images) in the form of numpy arrays with shape (H, W, C).
          audio (numpy.ndarray): A 1D numpy array representing the audio signal corresponding to the video segment.

        Returns:
          dict: A dictionary containing:
              - 'track' (list of dict): The original tracking information passed as input.
              - 'proc_track' (dict): key: 'x', 'y', 's', representing facebox center and half-size; value: list of floats. 
              - 'data' (list): A list containing: 
                a. crop_frames (list of numpy.ndarray): Cropped face clips resized to (224, 224, 3); 
                b. cropaudio (numpy.ndarray): Cropped audio segment corresponding to this track, of shape (n_frames*self.asvf_ratio,).

        Notes:
          - The function pads the frames to ensure the bounding boxes do not go out of bounds.
          - The `self.crop_scale` parameter determines the padding scale for the bounding boxes.
          - The `self.asvf_ratio` parameter is used to map frame indices to audio samples.
        """
        ## get face center and size info of each frame in the track
        crop_frames = []
        dets = {'x':[], 'y':[], 's':[]}
        for det in track['bbox']:
            dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) # half crop box size
            dets['y'].append((det[1]+det[3])/2) # crop center y
            dets['x'].append((det[0]+det[2])/2) # crop center x
        
        ## crop face from each frame
        for fidx, frame in enumerate(track['frame']):
            cs  = self.crop_scale
            bs  = dets['s'][fidx]   # detection box size
            bsi = int(bs * (1 + 2 * cs))  # pad videos by this amount 
            ### pad bsi pixels for each side of the image to avoid out of boundary when cropping
            image = frames[frame]
            frame = np.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
            ### get crop box center in padded image
            my  = dets['y'][fidx] + bsi  # BBox center Y
            mx  = dets['x'][fidx] + bsi  # BBox center X
            ### crop and resize face
            face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
            crop_frames.append(cv2.resize(face, (224, 224)))
        
        ## crop corresponding audio segment
        cropaudio = audio[int(track['frame'][0]*self.asvf_ratio): int((track['frame'][-1]+1)*self.asvf_ratio)]
        return {'track':track, 'proc_track':dets, 'data':[crop_frames, cropaudio]}

    def evaluate_asd(self, tracks):
        """
        Use pretrained TalkNet to detect active speakers in each face track in tracks.

        Args:
          tracks (list): of length len(allTracks), each element is a dict contains:
            - 'track' (dict): current face track in the form of dictionary with the following keys:
              a. 'frame' (numpy.ndarray): A 1D array of continuous frame indices corresponding to the track.
              b. 'bbox' (numpy.ndarray): A 2D array of shape (num_frames, 4), containing the interpolated bounding boxes for each frame in the track.
            - 'proc_track' (dict): key: 'x', 'y', 's', representing facebox center and half-size; value: list of floats. 
            - 'data' (list): A list containing: 
              a. crop_frames (list of numpy.ndarray): Cropped face clips resized to (224, 224, 3); 
              b. cropaudio (numpy.ndarray): Cropped audio segment corresponding to this track, of shape (n_frames*self.asvf_ratio,).

        Returns:
          all_scores (list): contains the active speaker scores for each track. Each element is a numpy array of shape (num_frames, ), representing the active speaker scores for each frame in the track.


        Notes:
          - Only data part in tracks is used in this function
          - Crop_frames are from original video, so need to convert to gray first
          - This function just copy from official TalkNet repo
        """        
        all_scores = []
        for ins in tracks:
            video, audio = ins['data']
            ## extract mel-spectrogram as audio feature
            audio_feature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010) # (n_audio_frames, 13)
            ## process video frames: convert to gray, resize to 224*224, and crop center 112*112
            video_feature = []
            for frame in video:
                face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224,224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                video_feature.append(face)
            video_feature = np.array(video_feature) # (n_video_frames, 112, 112)
            ## calculate min time length(s) of audio/video
            timestep_per_frame_audio = 0.010
            timestep_per_frame_video = 1 / 25
            ### audio_feature.shape[0] should be multiple of 4, since one video frame corresponds to 4 audio frames
            time_len_audio = (audio_feature.shape[0] - audio_feature.shape[0] % 4) * timestep_per_frame_audio
            time_len_video = video_feature.shape[0] * timestep_per_frame_video
            length = min(time_len_audio, time_len_video)
            ## align time length of audio and video
            audio_feature = audio_feature[:int(round(length / timestep_per_frame_audio)),:]
            video_feature = video_feature[:int(round(length / timestep_per_frame_video)),:,:]
            audio_feature = np.expand_dims(audio_feature, axis=0).astype(np.float32)
            video_feature = np.expand_dims(video_feature, axis=0).astype(np.float32)
            ## get active speaker score
            score = self.speaker_detector(audio_feature, video_feature)
            all_score = np.round(score, 1).astype(float)
            all_scores.append(all_score)	
        return all_scores

    def evaluate_fr(self, frames, tracks, scores):
        """
        Extract high-quality facial embeddings 
        Args:
          frames (list of numpy.ndarray): A list of original video frames (images) in the form of numpy arrays with shape (H, W, C).          
          tracks (list): of length len(allTracks), each element is a dict contains:
            - 'track' (dict): current face track in the form of dictionary with the following keys:
              a. 'frame' (numpy.ndarray): A 1D array of continuous frame indices corresponding to the track.
              b. 'bbox' (numpy.ndarray): A 2D array of shape (num_frames, 4), containing the interpolated bounding boxes for each frame in the track.
            - 'proc_track' (dict): key: 'x', 'y', 's', representing facebox center and half-size; value: list of floats. 
            - 'data' (list): A list containing: 
              a. crop_frames (list of numpy.ndarray): Cropped face clips resized to (224, 224, 3); 
              b. cropaudio (numpy.ndarray): Cropped audio segment corresponding to this track, of shape (n_frames*self.asvf_ratio,).       
          scores (list): contains the active speaker scores for each track. Each element is a numpy array of shape (num_frames, ), representing the active speaker scores for each frame in the track.

        Returns:
          active_facial_embs (dict): contains:
            - 'frameI' (numpy.ndarray): A 1D array of frame indices where only one active face was detected and its quality score was above 0.7. frame indices are in range(len(frames)).
            - 'feat' (numpy.ndarray): A 2D array of shape (num_faces, 512), where each row represents the facial embedding of an active face.

        Notes:
          - The function filters out frames with multiple active faces or low-quality faces. Only frames with a single active face and a face quality score above 0.7 are used for embedding extraction.
          - The embeddings are extracted using a pretrained CurricularFace model.

        """              
        faces = [[] for i in range(len(frames))]  # each element is a list, containing all tracked faces and their scores in the corresponding frame.
        # Aggregate active speaker scores and cropped faces in each frame of all tracks to faces(list)
        for tidx, track in enumerate(tracks):
            ## get active speaker score of each frame in the track
            score = scores[tidx]
            ## fidx-->index of frame in the track; frame--> index of the frame in original video segment
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                ### get smoothed active speaker score for current frame
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
                s = np.mean(s)
                ### get cropped face in original video
                bbox = track['track']['bbox'][fidx]
                face = frames[frame][max(int(bbox[1]), 0):min(int(bbox[3]), frames[frame].shape[0]), max(int(bbox[0]), 0):min(int(bbox[2]), frames[frame].shape[1])]
                ### save
                faces[frame].append({'track':tidx, 'score':float(s), 'facedata':face})

        # For each frame processed by face detection
        active_facial_embs={'frameI':np.empty((0,), dtype=int), 'feat':np.empty((0, 512), dtype=np.float32)}
        for fidx in range(len(faces)):
            if fidx % self.face_det_stride != 0:
                continue
            active_face = None
            ## count the number of active faces in the frame and keep active_face
            active_face_num = 0
            for face in faces[fidx]:
                if face['score'] > 0:
                    active_face = face['facedata']
                    active_face_num += 1
            ## extract facial embedding from frames containing only one active face, and the face quality must be good enough
            if active_face_num == 1:
                ### require face quality to be good enough
                face_quality_score = self.face_quality_evaluator(active_face)
                if face_quality_score < 0.7:
                    continue
                ### use CurricularFace to extract facial embedding
                feature = self.face_embs_extractor(active_face)
                active_facial_embs['frameI'] = np.append(active_facial_embs['frameI'], fidx)
                active_facial_embs['feat'] = np.append(active_facial_embs['feat'], feature, axis=0)
        return active_facial_embs

    def visualization(self, frames, tracks, scores):
        faces = [[] for i in range(len(frames))]
        for tidx, track in enumerate(tracks):
            score = scores[tidx]
            for fidx, frame in enumerate(track['track']['frame'].tolist()):
                s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]
                s = np.mean(s)
                faces[frame].append({'track':tidx, 'score':float(s),'bbox':track['track']['bbox'][fidx]})

        colorDict = {0: 0, 1: 255}
        for fidx, image in enumerate(frames):
            for face in faces[fidx]:
                clr = colorDict[int((face['score'] >= 0))]
                txt = round(face['score'], 1)
                cv2.rectangle(image, (int(face['bbox'][0]), int(face['bbox'][1])), (int(face['bbox'][2]), int(face['bbox'][3])),(0,clr,255-clr),10)
                cv2.putText(image,'%s'%(txt), (int(face['bbox'][0]), int(face['bbox'][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
            self.v_out.write(image)
