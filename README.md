# TrashTalk
멀티모달 딥러닝 기반 생활폐기물 분류 및 분리배출 안내 서비스

## 팀원
정주영, 김경남, 송승원, 김용희, 김형균

## 프로젝트 개요
EfficientNet-B0(이미지) + klue/roberta-base(텍스트) Cross-Attention 융합으로
23개 생활폐기물 클래스 분류 및 환경부 기준 분리배출 안내 제공

## 모델
- Hugging Face: https://huggingface.co/huggingleg12/BertEfficient_recycle

## 폴더 구조
- `preprocessing/` : 데이터 전처리 코드
- `modeling/` : 모델 학습 및 비교 분석 코드
- `service/` : 분류 API 서버 및 프론트엔드
