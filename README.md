# Global AI Ethics Simulator — T3

인공지능과 경영 11주차 협동학습 · 배정 문화권: `MIDDLE_E`

## 현재 상태

이 저장소는 강의에서 제공한 팀 템플릿을 복사한 **작업 시작본**입니다.

- `data/scenarios.json`은 제공된 **KOREA 예시 데이터**입니다. T3의 중동 시나리오가 완성되면 교체합니다.
- `app.py`는 아직 예시 시나리오를 코드 안에서 사용합니다. T3 JSON을 읽도록 연결해야 합니다.
- `narrative_app.py`는 구현 안내만 담긴 시작 파일입니다.
- 보고서와 AI 사용 로그의 예시 문장은 실제 조사·실험 결과로 교체해야 합니다.

## 팀 정보

| 항목 | 내용 |
| --- | --- |
| 팀 | T3 |
| 배정 문화권 | MIDDLE_E |
| 조사 대상 국가·지역 | 미정 |
| 최종 Streamlit URL | 배포 후 기입 |
| 최종 제출일 | 확인 후 기입 |

| 이름 | GitHub ID | 담당 역할 |
| --- | --- | --- |
| 기입 예정 | 기입 예정 | 기입 예정 |

## 최종 산출물

- `docs/week1/culture_report.md`: 조사 근거, 가중치, 검증 가능한 가설
- `data/scenarios.json`: T3의 중동 딜레마 시나리오 3개 이상
- `docs/week2/scenario_review.md`: 보상값 근거, 다른 팀 리뷰와 대응
- `docs/week3/final_report.md`: 실행 조건, 지표, 그래프, 감사, 재설계
- `logs/ai_usage_log.md`: AI 사용 전 초안, 프롬프트, 응답, 팀 수정과 판단 근거
- `logs/experiments/`: 시드별 CSV와 최종 그래프
- GitHub PR·Issue 기록과 Streamlit 배포 URL

## 로컬 실행

Python 3.11을 기준으로 합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/validate_scenarios.py --file data/scenarios.json --schema schemas/scenario.schema.json
streamlit run app.py
```

현재 검증 명령은 제공된 KOREA 예시 JSON을 검사합니다. T3 데이터를 작성한 후 같은 명령으로 다시 검증합니다.

## 협업 규칙

1. 각자 작업 브랜치를 만들고 PR로 `main`에 병합합니다.
2. 매 단계에서 AI 사용 전 초안을 먼저 남기고 `logs/ai_usage_log.md`에 변경 이유를 기록합니다.
3. 출처 URL, 시드, 실험 파라미터와 결과 파일을 함께 보관합니다.
4. 최종 제출 전 새 clone에서 설치·검증·앱 실행·실험 재현을 확인합니다.

제공 자료: 강송희 교수, *Global AI Ethics Simulator* 학생 패키지, CC BY-NC 4.0.
