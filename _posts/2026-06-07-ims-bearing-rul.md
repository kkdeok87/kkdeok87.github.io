
---
title: "NASA IMS Bearing Dataset 기반 Bearing RUL Prediction"
date: 2026-06-07
categories: [AI, Predictive Maintenance]
tags: [RUL, Bearing, LSTM, Deep Learning, Predictive Maintenance]
---

# NASA IMS Bearing Dataset 기반 Bearing RUL Prediction

## 1. 프로젝트 개요

산업 설비의 예지보전(Predictive Maintenance) 분야에서는 단순히 고장 여부를 판별하는 것보다 앞으로 얼마나 사용할 수 있는지 예측하는 Remaining Useful Life(RUL) 추정이 중요하다.

이번 프로젝트에서는 NASA IMS Bearing Dataset을 활용하여 베어링 진동 데이터를 분석하고, Remaining Useful Life(RUL)를 예측하는 딥러닝 모델을 구현하였다.

최종 목표는 특정 시점의 진동 데이터로부터 남은 수명을 시간 단위로 추정하는 것이다.

---

## 2. 데이터셋 소개

NASA IMS Bearing Dataset은 University of Cincinnati IMS Center에서 수행한 가속 수명 시험 데이터를 제공한다.

실험 조건은 다음과 같다.

* Sampling Rate: 20 kHz
* Measurement Duration: 1 second
* Measurement Interval: 10 minutes
* Rotational Speed: 2000 RPM
* Bearing Test Rig 기반 가속 수명 시험

실험은 총 3개의 테스트 셋으로 구성되어 있다.

| Dataset | 용도    |
| ------- | ----- |
| Set1    | Train |
| Set2    | Train |
| Set3    | Test  |

Set1과 Set2에는 고장이 발생한 베어링 채널이 명확하게 정의되어 있으며, Set3은 최종 성능 평가에 사용하였다.

---

### 데이터 구조

[이미지 삽입 예정]

```text
CSV File
 ├─ Channel 0
 ├─ Channel 1
 ├─ Channel 2
 └─ Channel 3

1 File = 1초 진동 데이터
1 Dataset = 수백~수천 개의 파일
```

---

## 3. 문제 정의

초기에는 베어링 고장 여부를 분류하는 Classification 문제를 고려하였다.

그러나 실제 산업 환경에서는 단순히 고장 여부보다

> "앞으로 얼마나 사용할 수 있는가?"

가 더욱 중요하다.

따라서 본 프로젝트는 Binary Classification 대신 RUL Regression 문제로 정의하였다.

입력:

* 진동 신호 특징

출력:

* Remaining Useful Life (시간)

---

## 4. Feature 탐색

원본 데이터는 시간 영역(Time Domain)의 가속도 값이다.

베어링 결함은 일반적으로 특정 주파수 대역의 에너지 증가로 나타나므로 주파수 영역 분석을 수행하였다.

---

### 4.1 FFT 기반 스펙트럼 분석

먼저 각 파일에 대해 FFT를 수행하여 진폭 스펙트럼을 확인하였다.

[이미지 삽입 예정]

고장 채널에서는 시간이 지남에 따라 고주파 성분이 증가하는 경향을 확인할 수 있었다.

---

### 4.2 Welch PSD

일반 FFT는 노이즈에 민감하므로 Welch PSD(Power Spectral Density)를 적용하였다.

Welch 방법은 신호를 여러 구간으로 나누어 FFT를 수행한 뒤 평균을 계산하여 보다 안정적인 주파수 특성을 제공한다.

---

### 4.3 최종 선택한 Feature

실험 결과 다음 3개의 Feature를 사용하였다.

#### Feature 1. AMP_like

PSD의 제곱근 값을 사용하였다.

```text
AMP_like = √PSD
```

직관적으로 진폭 크기를 표현한다.

---

#### Feature 2. PSD

주파수별 에너지 밀도를 표현한다.

베어링 열화가 진행될수록 특정 주파수 대역의 PSD가 증가하는 경향을 보인다.

---

#### Feature 3. Band Energy

주파수 대역별 에너지를 계산하였다.

사용한 대역은 다음과 같다.

| Band | Frequency     |
| ---- | ------------- |
| B1   | 0~200 Hz      |
| B2   | 200~800 Hz    |
| B3   | 800~2000 Hz   |
| B4   | 2000~5000 Hz  |
| B5   | 5000~10000 Hz |

---

### Feature 변화 추세

특히 5~10 kHz 대역에서 열화 징후가 가장 뚜렷하게 나타났다.

[이미지 삽입 예정]

---

## 5. RUL Label 생성

실험 종료 시점을 Failure Point로 정의하였다.

각 시점의 RUL은 다음과 같이 계산하였다.

```text
RUL = Failure Time - Current Time
```

---

### Capped RUL

초기 데이터 대부분은 정상 상태이므로 지나치게 큰 RUL 값을 가진다.

이를 방지하기 위해 RUL을 24시간으로 제한하였다.

```text
RUL_capped = min(RUL, 24h)
```

[이미지 삽입 예정]

이 방법을 통해 모델이 실제 열화 구간에 집중하도록 유도하였다.

---

## 6. 시퀀스 윈도우 구성

LSTM은 시계열 패턴을 학습하기 때문에 단일 시점이 아니라 연속된 데이터가 필요하다.

본 프로젝트에서는

* Window Length = 12
* Stride = 1

을 사용하였다.

즉,

```text
12개 측정 파일
≈ 약 2시간
```

의 변화 패턴을 입력으로 사용하였다.

---

## 7. LSTM 모델 설계

모델 구조는 다음과 같다.

```text
Input
  ↓
LSTM(128)
  ↓
Dense(64)
  ↓
Dense(1)
```

출력은 특정 시점의 Remaining Useful Life이다.

[모델 구조 이미지 삽입 예정]

---

## 8. 학습 및 검증

학습 데이터:

* Set1
* Set2

검증 데이터:

* Train Set 내부 마지막 20%

테스트 데이터:

* Set3

---

## 9. 테스트 결과

최종 모델은 Set3 데이터 중 10% 랜덤 샘플링된 시퀀스 윈도우를 이용하여 평가하였다.

### 전체 테스트 결과

| Metric | Result   |
| ------ | -------- |
| MAE    | 0.2665 h |
| RMSE   | 2.0367 h |
| MAPE   | 27.77 %  |

---

### Near Failure 구간

실제 고장에 가까운 하위 100개 샘플에 대해 평가하였다.

| Metric | Result   |
| ------ | -------- |
| MAE    | 1.5418 h |
| RMSE   | 5.1157 h |
| MAPE   | 174.68 % |

---

## 10. 결과 해석

전체 테스트 구간에서는 약 24시간 범위의 RUL 예측 문제에 대해 RMSE 약 2시간 수준의 성능을 달성하였다.

이는 모델이 평균적으로 약 ±2시간 수준의 오차 범위 내에서 남은 수명을 추정할 수 있음을 의미한다.

반면 고장 직전 구간에서는 오차가 증가하였다.

이는

* 학습 데이터 불균형
* 급격한 열화 패턴
* 매우 작은 RUL 값

등의 영향으로 판단된다.

---

## 11. 결론

본 프로젝트에서는 NASA IMS Bearing Dataset을 활용하여 Bearing Remaining Useful Life Prediction 모델을 구현하였다.

단순 고장 분류 대신 RUL Regression 문제로 접근하였으며,

* Welch PSD
* AMP_like
* Band Energy

기반 Feature를 활용하여 베어링 열화 특성을 추출하였다.

최종적으로 Set3 테스트 데이터에서 RMSE 약 2시간 수준의 성능을 달성하였으며, 향후 TCN, Attention, Transformer 기반 모델과 비교 실험을 수행할 예정이다.
