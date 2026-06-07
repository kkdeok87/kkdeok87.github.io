---
title: "NASA IMS Bearing Dataset 기반 Bearing RUL Prediction"
date: 2026-06-06
categories: [AI, Predictive-Maintenance]
tags: [RUL, Bearing, LSTM, Deep-Learning]
author: "강경덕, 장희중, 김기훈, 박창진"
---

# NASA IMS Bearing Dataset 기반 Bearing RUL Prediction

> **AX: 딥러닝 G24**
>
> 공동 연구자:
> 강경덕, 장희중, 김기훈, 박창진

## 프로젝트 개요

예지보전(Predictive Maintenance)에서는 단순히 설비의 고장 여부를 판단하는 것보다 앞으로 얼마나 더 사용할 수 있는지 예측하는 것이 중요하다.

이번 프로젝트에서는 NASA IMS Bearing Dataset을 활용하여 베어링 진동 데이터를 분석하고, Remaining Useful Life(RUL)를 예측하는 딥러닝 모델을 구현하였다.

최종 목표는 특정 시점의 진동 데이터로부터 베어링의 남은 수명을 시간 단위로 추정하는 것이다.

---

## 데이터셋 소개

NASA IMS Bearing Dataset은 베어링 가속 수명 시험 데이터를 제공한다.

실험 조건은 다음과 같다.

* Sampling Rate : 20 kHz
* Measurement Duration : 1 second
* Measurement Interval : 10 minutes
* Rotational Speed : 2000 RPM

각 CSV 파일은 1초 동안 측정된 진동 신호를 저장하고 있으며, 여러 파일이 시간 순서대로 저장되어 베어링 열화 과정을 관찰할 수 있다.

---

## Feature 탐색

원본 데이터는 시간 영역(Time Domain)의 진동 데이터이다.

베어링 결함은 특정 주파수 대역의 에너지 증가로 나타나는 경우가 많기 때문에 주파수 영역 분석을 수행하였다.

---

### 1. Amplitude Spectrum

먼저 FFT를 이용하여 진동 데이터를 주파수 영역으로 변환하였다.

![Amplitude Spectrum](/assets/img/Amplitude_Spectrum.png)

진폭 스펙트럼을 통해 특정 주파수 성분이 얼마나 강하게 나타나는지 확인할 수 있다.

---

### 2. Power Spectral Density (PSD)

FFT 결과는 노이즈의 영향을 받을 수 있기 때문에 Welch PSD를 이용하여 보다 안정적인 주파수 특성을 계산하였다.

![Power Spectral Density](/assets/img/Power_Spectral_Density.png)

PSD는 특정 주파수 대역에 얼마나 많은 에너지가 분포하는지를 나타낸다.

---

### 3. Band Energy

PSD 전체를 사용하는 대신 주파수 구간별 에너지를 계산하였다.

사용한 대역은 다음과 같다.

| Band | Frequency Range |
| ---- | --------------- |
| B1   | 0 ~ 200 Hz      |
| B2   | 200 ~ 800 Hz    |
| B3   | 800 ~ 2000 Hz   |
| B4   | 2000 ~ 5000 Hz  |
| B5   | 5000 ~ 10000 Hz |

아래 그래프는 채널 0의 주파수 대역별 에너지 변화 예시이다.

![Band Energy](/assets/img/Band_Energy_Channel_0.png)

---

## RUL Label 생성

초기에는 고장 여부를 분류하는 Classification 문제를 고려하였다.

하지만 실제 산업 환경에서는 "고장이 났는가?" 보다 "앞으로 얼마나 사용할 수 있는가?" 가 더욱 중요하다.

따라서 이번 프로젝트에서는 RUL Regression 문제로 정의하였다.

각 시점의 RUL은 다음과 같이 계산하였다.

```text
RUL = Failure Time - Current Time
```

또한 정상 구간이 지나치게 길어지는 문제를 줄이기 위해 RUL을 최대 24시간으로 제한하였다.

```text
RUL_capped = min(RUL, 24h)
```

---

## 모델 구성

모델 입력은 다음과 같다.

* Amplitude Spectrum
* Power Spectral Density
* Band Energy

각 Feature를 정규화한 뒤 시계열 형태로 구성하였다.

* Window Length : 12
* Stride : 1

즉 약 2시간 동안의 진동 특성 변화를 하나의 입력 시퀀스로 사용하였다.

모델은 LSTM 기반 회귀 모델을 사용하였다.

```text
Input
 ↓
LSTM(128)
 ↓
Dense(64)
 ↓
Dense(1)
```

출력값은 해당 시점의 Remaining Useful Life(RUL)이다.

---

## 결과

Set1과 Set2 데이터를 이용하여 학습을 수행하고, Set3 데이터를 이용하여 최종 성능을 평가하였다.

![RUL Result](/assets/img/RUL_결과.PNG)

최종 테스트 결과는 다음과 같다.

| Metric | Result   |
| ------ | -------- |
| MAE    | 0.2665 h |
| RMSE   | 2.0367 h |
| MAPE   | 27.77 %  |

고장 직전 데이터(하위 100개 샘플)에 대해서는 다음 결과를 얻었다.

| Metric | Result   |
| ------ | -------- |
| MAE    | 1.5418 h |
| RMSE   | 5.1157 h |
| MAPE   | 174.68 % |

---

## 결과 해석

CAP 24시간 조건에서 전체 테스트 데이터에 대해 약 2시간 수준의 RMSE를 달성하였다.

이는 모델이 남은 수명 24시간 범위 내에서 평균적으로 약 ±2시간 수준의 오차로 베어링 수명을 예측할 수 있음을 의미한다.

반면 고장 직전 구간에서는 오차가 증가하였다.

이는 실제 고장에 가까워질수록 열화 패턴이 급격하게 변화하고, 상대적으로 학습 데이터 수가 적기 때문으로 판단된다.

---

## 마무리

이번 프로젝트를 통해 진동 신호 기반의 Remaining Useful Life(RUL) 예측 과정을 처음부터 구현해보았다.

특히 FFT, PSD, Band Energy와 같은 주파수 영역 Feature가 베어링 열화 패턴을 표현하는 데 효과적임을 확인할 수 있었으며, LSTM 기반 모델을 통해 실제 수명 예측 문제를 경험할 수 있었다.

향후에는 TCN(Temporal CNN), Attention, Transformer 기반 모델과의 성능 비교도 수행해 볼 예정이다.

---

## Related Works

- NASA IMS Bearing Dataset
  - https://data.nasa.gov/dataset/ims-bearings

- Kaggle Bearing Dataset
  - https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
