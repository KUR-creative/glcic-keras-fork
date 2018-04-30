 [연구(가장 먼저)]
1.
complnet만 따로 저장하거나, (save_weight/load_weight로 가능한 듯.)
입력단만 바꿔 저장하거나...
어떻게 해서든 variable 크기의 입력을 가능하게 하기.
    1 - mnist로 일부 웨이트만 세이브/로드 시험해보기
    2 - glcic로 해보기

mse까지만 학습시켜서 test time에 여러 사이즈 되는지 확인(작은 dataset으로 확인)
    3 - 학습시에는 128x128로, 테스트 시에는 여러 사이즈로.
    ㄴ 인터렉티브 테스터 만들기.
    ㄴ 이밸류에이터, eval dataset 만들기

 [more]
2.
만화 이미지를 128x128로 만들기
만든 이미지들로 학습시키기.

real-time augmentation
    다양한 것이 가능하게 하기..
배치 생성 병렬화하기 - 배치 생성에 워커n개 / 학습(gpu)에 1개 

텐서보드 지원
easy setting - 커맨드라인 인자 지원하기.
-------------------------------------
모든 기능은 시간을 재어 최적화하기
데이터 프로세싱은 함수형으로

use GIT PROPERLY!
boyscout rule!

it's PRODUCT level code.
refactor!
clean name!
    느긋하게, 하지만 끈질기게!
    천천히, 하지만 꾸준히!
    TEST! TEST! TEST!
