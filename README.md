# BIG DATA Project_A1
for BIG DATA Project communication and collaboration

- 2021/05/27 gitkraken을 이용한 관리

# AI Project_A1

## 6/16

### **video captioning**

1. 전기수(8기) 연구인턴 참고. [github link](https://github.com/Hinterhalter/CCTV_Video_Captioning)
2. 그 중 [grounded-video-description](https://github.com/facebookresearch/grounded-video-description) 참고 
3. clone 후 conda create 해줌
   - conda env create -f cfgs/conda_env_gvd_py3.yml
   - 실행 시 gvd_py3_pytorch1.1 의 conda 생성됨 (pytorch1.1)
4. clone 한 repo에는 누락된 사항이 있음
   - tools/densevid_eval 채워주기 [링크](https://github.com/LuoweiZhou/densevid_eval_spice)
   - tools/densevid_eval/coco-caption 채워주기 [링크](https://github.com/LuoweiZhou/coco-caption/tree/de6f385503ac9a4305a1dcdc39c02312f9fa13fc)
5. pretrained models 다운로드 및 save 폴더에 저장 (Pre-trained Models 탭) 
   - https://dl.fbaipublicfiles.com/ActivityNet-Entities/ActivityNet-Entities/pre-trained-models.tar.gz
6. 전기수가 2.27 언급한 사항 중 main.py에 eval함수가 사용되지 않은 오류
   - 전기수github[링크](https://github.com/Hinterhalter/CCTV_Video_Captioning) -> flicker/main.py 다운로드 후 대체
7. eval_grd_flickr30k_entities.py가 없다는 오류로  etools/anet_entities 경로에 다운받아줌. [링크](https://github.com/facebookresearch/grounded-video-description/tree/flickr_branch)
   - 본 깃헙에 flicker_branch 를 ctrl+F 하면 찾을 수 있음

현재 오류
- Exception: only support flickr30k!

남은 문제들
- 직접 train 하기에는 data 크기가 216GB로 용량부족하고 , computing power 도 부족함 
- 우선 inference model 돌려보고 오류사항 계속 체크할 것

## 6/17

### **video captioning**

1. 데이터가 두 분류 나뉘는 것을 확인
   - ActivityNet-Entities / flickr30k
2. 원본 레포지토리는 anet 을 이용. flickr 를 이용한 branch를 clone 해봄
   - git clone -b flickr_branch --recursive https://github.com/facebookresearch/grounded-video-description
3. flickr는 image data로 training, anet 은 영상 포함 -> 전기수는 flickr로 함
4. flickr의 Pre-trained Models를 받아서 save 밑에 위치시킴
5. Inference 시 없는 파일이 계속해서 존재
   - Data Preparation 에서 찾아서 다운로드함. data/dic_anet.json 등
   - dic_flickr는 찾지 못함

현재 오류
- 정확한 Inference 실행 코드 찾지못함 (arguments 설정값 찾아야 함)
- 찾지 못한 필요 파일들 찾기
- Issues 뒤져보기 (train 말고)

남은 오류
- Inference가 내 비디오를 돌려서 나오는건지.. visualization이 필요. 가야할 길 멀다

### **Text To Speech**
1. LPCtron (tacotron2 + LPCNet) 모델 사용 [github link](https://github.com/alokprasad/LPCTron)
2. conda create 후 sh tts.sh을 통해 test 완료
3. train을 위해 다음 명령어 시행
```python3 preprocess.py --base_dir /media/alok/ws/sandbox/lpc_tacatron2/dataset --dataset LJSpeech-1.1```
- 근데 짜증
- 패키지 설치 똑같은 거 100번 중
- 설치 방금했는데 또하고 또하고 또하고 또함
- base랑 가상환경이랑 따로 설치되는거 같은데 공부하기 싫고 걍 100번 설치 하는 중
- 설치하다가 퇴근 시간 돼서 퇴근할거임

## 6/18
**librosa 나쁜놈**     
희란언니를 괴롭게 하다니       
내가 오늘 librosa 혼쭐낸다   
양꼬치 먹으러 즐겁게 다녀오셔요   
혼쭐낸다는 녀석은 혼쭐났다고 한다.... :(     
알맞게 파악한지는 모르겠지만 librosa가 없다고 한 이유는 정말 없어서였다고 한다.     
import sys를 해서 어디서 package를 가져오나 봤더니 local에서 가져오는거 같았는데    
pip show librosa해서 보니까 가상환경에 깔려있었다.    
sys.path.append()해서 가상환경의 패키지들이 있는 위치를 넣어주었더니 오류가 해결!  

된 줄 알았는데 또 다른 오류 발생.     
눈물날 것 같아서 이만 로그를 그만    

쓰고 싶었는데 또 쓴다.
다음 Error를 검색해보니 tensorflow와 protobuf에서 문제가 생긴듯 하였다.
근데 이걸 해결하는 과정에서 무언가 단단히 꼬인듯,,
원래 꼬인 문젠지 내가 꼬아 놓은 문젠지 확신이 가지 않는다
환경설정이 넘모너무 싫다    
- 이만 퇴근하겠다! 

## 6/19
**달리즈아아** => 세시출근..   
하루종일 tensorflow와 protobuf에 시달렸다..   
결론적으로 말하면 해결을 못하고 새로운 모델을 구했다    
**천재 희란 짱짱맨**

### **Text To Speech**
1. LPCtron에 librosa는 지우디우가 해결해 줬음
2. 그런데 serialized_options 오류로 하루종일 개고생함
3. tensorflow 랑 protobuf 버전 문제인거 같은데 해결 못함

4. 그래서 모델 바꿈
5. tacotron2 + waveglow 모델 
   - 이거는 2021년 4월 모델임
   - 괜히 옛날 모델 갖고 난리난리였던거 같음

   
     

[참고 블로그](https://joungheekim.github.io/2021/04/02/code-review/)          
[원본 tacotron2 git link](https://github.com/NVIDIA/tacotron2)     
   - package 설치 및 train에 도움 받을 것    
        
        
[한국어 tacotron2 git link](https://github.com/JoungheeKim/tacotron2)     
[hccho2 git link](https://github.com/hccho2/Tacotron2-Wavenet-Korean-TTS)      
   - 크게 도움 받을 일은 없으나 우선 참고

앞으로 해야할 일
1. kss data 전처리 (참고 블로그 참고)
2. model train
3. waveglow 모델

### **video captioning**
1. Flickr에서 필요한 feature 데이터들을 전체 다운로드 했다. (Anet은 216GB라 하지 못함)
2. command 우선 확정 (visualization X)
```
python main.py --path_opt cfgs/flickr30k_res101_vg_feat_100prop.yml --batch_size 50 --cuda --num_workers 10 --max_epoch 50 --inference_only --start_from save/flickr-sup-0.1-0.1-0.1-run1 --id flickr-sup-0.1-0.1-0.1-run1 --val_split test --seq_length 20 --language_eval --eval_obj_grounding --obj_interact
```
3. 하찮은 영어로 에러에 대한 question issue 생성 [링크](https://github.com/facebookresearch/grounded-video-description/issues/37) but 웬만큼 해결되어 지웠다 ㅎㅎ
4. 전기수분한테도 문의메일 보내놓음
5. inference my own video 과정에 대해 학습함 
- sampling the video (비디오를 프레임을 쪼갠다. 비디오당 10frame만 했다고 한다.)
- calculate the features of the sampled frames (프레임→feature extraction)
 - Region features: can be obtained using extract_features.py and Detectron
 - Frame-wise features : I have no idea how to calculate them
- use code for inference (드디어 inference code를 쓸 수 있게 된다.)
**목표가 inference 라면 caption annotations는 필요없다** = 당연
6. 꼬인게 있을 수 있어 다 지우고 다시 시작


현재 에러
- 실행 시 data/flickr30k/flickr30k_detection_vg_X-101-64x4d-FPN_2x_feature/6827875949.npy 가 없다고 뜬다. 여기서 npy 파일은 eval 로 split 된 이미지 중 하나인데 실제로  data/flickr30k/flickr30k_detection_vg_X-101-64x4d-FPN_2x_feature 폴더의 npy 파일들을 dic_flickr.json 에서 확인해 봤을 때에는 train split 이미지의 feature extraction 된 npy 파일만 존재하는 걸 볼 수 있었다.

추후 할거
visualization / own video 로 inference 

## 6/20
희란언니가 training에 성공했다.     
이제 더 깊은 모델 공부가 필요하다.     
윤수는 전기수로부터 답 메일을 받았다.     
주제는 훌륭하다고 하지만, 확실히 비디오 캡셔닝이 큰 일이라고 말씀하셨다.    
윤수,,화이팅!      

## 6/20
### **Text To Speech**
오늘 본 에러 리스트(기억나는)

1. 수많은 import error
2. apex 관련
3. code is too big    

2번이랑 3번 에러 때문에 cuda version 확인하고 nvcc 깔고 cuda toolkit 설치하고 날려먹고 난리난리 개난리 nvidia-smi 안되는 순간부터 살고 싶지 않았음. 설치하면서 내가 이걸 왜 설치하나 현타옴.   
어찌저찌 설치해서ㅠㅠㅠㅠ 멀티코어로 code is too big 잡아냄ㅜㅠㅡㅠㅡㅜ batch_size 크게 하니까 됐음ㅠㅠㅠ 하루종일 batch_size 줄이기만 했는데ㅠㅠㅜㅜㅡㅠ     
어쨋든 돌아가서 통채로 깃에 올려버림

내일 할일    
1. train/validation 나누기
2. epoch 정하기
3. train 하기
4. train 결과가 뭔지 알아내기 및 저장하기(NLP 과제 참고하기)
5. Waveglow 모델 공부하기 및 돌리기
6. 기타 등등... 이제 생각안남

할게 많은데 정확하게 listup이 안된다. 내일 다시 써야징
