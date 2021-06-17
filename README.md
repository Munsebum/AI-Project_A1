# BIG DATA Project_A1
for BIG DATA Project communication and collaboration

- 2021/05/27 gitkraken을 이용한 관리

# AI Project_A1

## 6/16

**video captioning**

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

**video captioning**

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






## 6/17

**Text To Speech**
1. LPCtron (tacotron2 + LPCNet) 모델 사용 [github link](https://github.com/alokprasad/LPCTron)
2. conda create 후 sh tts.sh을 통해 test 완료
3. train을 위해 다음 명령어 시행
```python3 preprocess.py --base_dir /media/alok/ws/sandbox/lpc_tacatron2/dataset --dataset LJSpeech-1.1```
- 근데 짜증
- 패키지 설치 똑같은 거 100번 중
- 설치 방금했는데 또하고 또하고 또하고 또함
- base랑 가상환경이랑 따로 설치되는거 같은데 공부하기 싫고 걍 100번 설치 하는 중
- 설치하다가 퇴근 시간 돼서 퇴근할거임



