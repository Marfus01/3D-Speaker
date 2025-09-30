```bash
33368@LAPTOP-CAP10UHD MINGW64 /d/wangchen/Research/tv_series_plus/3D-Speaker (main)
$ conda activate 3D-speaker
(3D-speaker)


33368@LAPTOP-CAP10UHD MINGW64 /d/wangchen/Research/tv_series_plus/3D-Speaker (main)
$ cd /d/wangchen/Research/tv_series_plus/3D-Speaker/egs/3dspeaker/speaker-diarization
(3D-speaker)


33368@LAPTOP-CAP10UHD MINGW64 /d/wangchen/Research/tv_series_plus/3D-Speaker/egs/3dspeaker/speaker-diarization (main)
$ bash run_video.sh
run_video.sh Stage 1: examples/video.list exists. Skip this stage.

run_video.sh Stage2: Prepare onnx files and extrack raw videos and audios...

run_video.sh Stage3: Extract audio speaker embeddings...
run_audio.sh Stage2: Do vad for input wavs...
2025-09-30 16:49:05,039 - modelscope - INFO - Use user-specified model revision: v2.0.4
Downloading Model from https://www.modelscope.cn to directory: C:\Users\33368\.cache\modelscope\hub\models\iic\speech_fsmn_vad_zh-cn-16k-common-pytorch
2025-09-30 16:49:06,140 - modelscope - INFO - Use user-specified model revision: v2.0.4
2025-09-30 16:49:06,224 - modelscope - INFO - initiate model from C:\Users\33368\.cache\modelscope\hub\models\iic\speech_fsmn_vad_zh-cn-16k-common-pytorch
2025-09-30 16:49:06,224 - modelscope - INFO - initiate model from location C:\Users\33368\.cache\modelscope\hub\models\iic\speech_fsmn_vad_zh-cn-16k-common-pytorch.
2025-09-30 16:49:06,227 - modelscope - INFO - initialize model from C:\Users\33368\.cache\modelscope\hub\models\iic\speech_fsmn_vad_zh-cn-16k-common-pytorch
funasr version: 1.2.7.
Check update of funasr, and it would cost few times. You may disable it by set `disable_update=True` in AutoModel
WARNING:root:trust_remote_code: False
2025-09-30 16:49:10,886 - modelscope - WARNING - No preprocessor field found in cfg.
2025-09-30 16:49:10,886 - modelscope - WARNING - No val key and type key found in preprocessor domain of configuration.json file.
2025-09-30 16:49:10,886 - modelscope - WARNING - Cannot find available config to build preprocessor at mode inference, current config: {'model_dir': 'C:\\Users\\33368\\.cache\\modelscope\\hub\\models\\iic\\speech_fsmn_vad_zh-cn-16k-common-pytorch'}. trying to build by task and model information.
2025-09-30 16:49:10,886 - modelscope - INFO - No preprocessor key ('funasr', 'voice-activity-detection') found in PREPROCESSOR_MAP, skip building preprocessor. If the pipeline runs normally, please ignore this log.
[INFO]: Start computing VAD...
rtf_avg: 0.016: 100%|███████████████████████████████████████████| 1/1 [00:00<00:00,  2.21it/s] 
[INFO]: VAD json is prepared in exp_video/json/vad.json
run_audio.sh Stage3: Prepare subsegments info...
[INFO]: Generate sub-segmetns...
[INFO]: Subsegments json is prepared in exp_video/json/subseg.json
run_audio.sh Stage4: Extract speaker embeddings...
Downloading Model from https://www.modelscope.cn to directory: C:\Users\33368\.cache\modelscope\hub\models\iic\speech_campplus_sv_zh_en_16k-common_advanced
2025-09-30 16:49:28,685 - modelscope - INFO - Use user-specified model revision: v1.0.0
[INFO]: Start computing embeddings...[WARNING]: The number of threads exceeds the number of files.
[WARNING]: The number of threads exceeds the number of files.[WARNING]: The number of threads exceeds the number of files.
D:\ProgramFiles\anaconda3\envs\3D-speaker\lib\site-packages\torch\nn\modules\conv.py:456: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\cudnn\Conv_v8.cpp:919.)
  return F.conv2d(input, weight, bias, self.stride,
D:\ProgramFiles\anaconda3\envs\3D-speaker\lib\site-packages\torch\nn\modules\conv.py:306: UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen\native\cudnn\Conv_v8.cpp:919.)
  return F.conv1d(input, weight, bias, self.stride,
  
run_video.sh Stage4: Extract visual speaker embeddings...
[WARNING]: Gpu 2 is not available. Use cpu instead.[WARNING]: Gpu 1 is not available. Use cpu instead.
[WARNING]: The number of threads exceeds the number of files.[WARNING]: The number of threads exceeds the number of files.
[WARNING]: Gpu 3 is not available. Use cpu instead.
[WARNING]: The number of threads exceeds the number of files.
[INFO]: Start computing visual embeddings...
video 7speakers_example info: w: 1920.0, h: 1080.0, count: 2201, fps: 25.0
2025-09-30 16:49:42.3475183 [E:onnxruntime:Default, provider_bridge_ort.cc:1992 onnxruntime::TryGetProviderInfo_CUDA] D:\a\_work\1\s\onnxruntime\core\session\provider_bridge_ort.cc:1637 onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126 "" when trying to load "D:\ProgramFiles\anaconda3\envs\3D-speaker\lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"
2025-09-30 16:49:42.3567187 [W:onnxruntime:Default, onnxruntime_pybind_state.cc:965 onnxruntime::python::CreateExecutionProviderInstance] Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*, and the latest MSVC runtime. Please install all dependencies as mentioned in the GPU requirements page (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), make sure they're in the PATH, and that your GPU is supported.       
2025-09-30 16:49:42.4855172 [E:onnxruntime:Default, provider_bridge_ort.cc:1992 onnxruntime::TryGetProviderInfo_CUDA] D:\a\_work\1\s\onnxruntime\core\session\provider_bridge_ort.cc:1637 onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126 "" when trying to load "D:\ProgramFiles\anaconda3\envs\3D-speaker\lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"
2025-09-30 16:49:42.4969269 [W:onnxruntime:Default, onnxruntime_pybind_state.cc:965 onnxruntime::python::CreateExecutionProviderInstance] Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*, and the latest MSVC runtime. Please install all dependencies as mentioned in the GPU requirements page (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), make sure they're in the PATH, and that your GPU is supported.       
2025-09-30 16:49:42.8590703 [E:onnxruntime:Default, provider_bridge_ort.cc:1992 onnxruntime::TryGetProviderInfo_CUDA] D:\a\_work\1\s\onnxruntime\core\session\provider_bridge_ort.cc:1637 onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126 "" when trying to load "D:\ProgramFiles\anaconda3\envs\3D-speaker\lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"
2025-09-30 16:49:42.8732639 [W:onnxruntime:Default, onnxruntime_pybind_state.cc:965 onnxruntime::python::CreateExecutionProviderInstance] Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*, and the latest MSVC runtime. Please install all dependencies as mentioned in the GPU requirements page (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), make sure they're in the PATH, and that your GPU is supported.       
2025-09-30 16:49:43.0478564 [E:onnxruntime:Default, provider_bridge_ort.cc:1992 onnxruntime::TryGetProviderInfo_CUDA] D:\a\_work\1\s\onnxruntime\core\session\provider_bridge_ort.cc:1637 onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126 "" when trying to load "D:\ProgramFiles\anaconda3\envs\3D-speaker\lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"
2025-09-30 16:49:43.0566364 [W:onnxruntime:Default, onnxruntime_pybind_state.cc:965 onnxruntime::python::CreateExecutionProviderInstance] Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*, and the latest MSVC runtime. Please install all dependencies as mentioned in the GPU requirements page (https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements), make sure they're in the PATH, and that your GPU is supported.       
The total processing time for 7speakers_example is 20.20s, including faceTime 2.57s, trackTime 0.02s, cropTime 5.24s, asdTime 3.16s, visTime 0.00s, featTime 9.21s.

run_video.sh Stage5: Clustering for both type of speaker embeddings...
[INFO] Start clustering...
[WARNING]: The number of threads exceeds the number of files
[WARNING]: The number of threads exceeds the number of files
[WARNING]: The number of threads exceeds the number of files

run_video.sh Stage6: Get the final metrics...
Computing DER...
Project root added to sys.path: D:\wangchen\Research\tv_series_plus\3D-Speaker
2025-09-30 16:50:30,468 - INFO: Concatenating individual RTTM files...
2025-09-30 16:50:30,551 - INFO: MS: 0.375888, FA: 0.670565, SER: 1.232076, DER: 2.278528
(3D-speaker) 
```

