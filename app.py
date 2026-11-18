# app.py — TU Korea AI Management: Ethical AI Simulation (with Strategy Entropy)
# Author: Prof. Songhee Kang
# Repository-optimized version: reproducibility, typing, English UI, stability improvements

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import random

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from scipy.stats import pearsonr, entropy as scipy_entropy


Action = Literal["A", "B"]
FRAMEWORKS: Tuple[str, ...] = ("emotion", "social", "moral", "identity")


# =========================
# 1. Page configuration
# =========================
st.set_page_config(
    page_title="AI 윤리 의사결정 시뮬레이터 | 3팀",
    page_icon="🎓",
    layout="wide",
)


# =========================
# 2. Environment model
# =========================
@dataclass(frozen=True)
class Scenario:
    """A single ethical dilemma scenario."""
    sid: str
    title: str
    setup: str
    options: Dict[Action, str]
    rewards: Dict[Action, Dict[str, float]]  # framework -> reward in [-1.0, 1.0]


@st.cache_data(show_spinner=False)
def get_default_scenarios() -> List[Scenario]:
    """Default scenarios used in the paper/demo."""
    return [
        Scenario(
            sid="S1",
            title="1단계: 고전적 트롤리 문제",
            setup="선로 위의 5명과 1명 사이에서 선택해야 합니다. 선로 전환 레버를 당기겠습니까?",
            options={"A": "1명을 희생해 5명을 구한다 (개입)", "B": "개입하지 않는다 (현상 유지)"},
            rewards={
                "A": {"emotion": 1.0, "social": -0.5, "moral": -1.0, "identity": 0.5},
                "B": {"emotion": -1.0, "social": 0.5, "moral": 1.0, "identity": -0.5},
            },
        ),
        Scenario(
            sid="S2",
            title="2단계: 상황적 요인",
            setup="무단출입자 5명과 시설 관리자 자녀 1명이 위험에 처했습니다.",
            options={"A": "5명을 구한다 (관리자 자녀 사망)", "B": "규칙을 따른다 (5명 사망)"},
            rewards={
                "A": {"emotion": 0.6, "social": -0.8, "moral": -0.7, "identity": 0.3},
                "B": {"emotion": -0.5, "social": 0.9, "moral": 0.6, "identity": 0.4},
            },
        ),
        Scenario(
            sid="S3",
            title="3단계: 의료 자원 배분",
            setup="다수의 부상 민간인과 숙련된 의사 1명 중 누구를 먼저 치료할지 결정해야 합니다.",
            options={"A": "의사를 우선 치료한다 (공리주의)", "B": "동등하게 치료한다 (평등주의)"},
            rewards={
                "A": {"emotion": 0.7, "social": -0.4, "moral": -0.6, "identity": 0.8},
                "B": {"emotion": -0.3, "social": 0.7, "moral": 0.9, "identity": 0.5},
            },
        ),
    ]


@st.cache_data(show_spinner=False)
def get_culture_presets() -> Dict[str, Dict[str, float]]:
    """Cultural weight presets (must sum to 1 after normalization)."""
    return {
        "USA":      {"emotion": 0.3, "social": 0.1, "identity": 0.3, "moral": 0.3},
        "CHINA":    {"emotion": 0.1, "social": 0.5, "identity": 0.2, "moral": 0.2},
        "EUROPE":   {"emotion": 0.3, "social": 0.2, "identity": 0.2, "moral": 0.3},
        "KOREA":    {"emotion": 0.2, "social": 0.2, "identity": 0.4, "moral": 0.2},
        "LATIN_AM": {"emotion": 0.4, "social": 0.2, "identity": 0.2, "moral": 0.2},
        "MIDDLE_E": {"emotion": 0.1, "social": 0.2, "identity": 0.2, "moral": 0.5},
        "AFRICA":   {"emotion": 0.2, "social": 0.4, "identity": 0.2, "moral": 0.2},
    }


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(weights.values())) or 1.0
    return {k: float(v) / total for k, v in weights.items()}


# =========================
# 3. Simple E-Greedy agent
# =========================
class SimpleEGreedyAgent:
    """
    A minimal E-Greedy reinforcement learning agent.

    - Maintains a Q-table per scenario (A/B).
    - Selects actions via epsilon-greedy exploration.
    - Updates Q-values using incremental mean style update:
        Q <- Q + lr * (reward - Q)
    """

    def __init__(
        self,
        name: str,
        weights: Dict[str, float],
        scenarios: List[Scenario],
        learning_rate: float = 0.1,
        epsilon: float = 0.5,
    ) -> None:
        self.name = name
        self.weights = weights
        self.scenarios = scenarios
        self.lr = float(learning_rate)
        self.epsilon = float(epsilon)

        self.q_table: Dict[str, Dict[Action, float]] = {
            s.sid: {"A": 0.0, "B": 0.0} for s in scenarios
        }

    def get_action(self, sid: str) -> Action:
        # 1) Exploration
        if random.random() < self.epsilon:
            return random.choice(["A", "B"])

        # 2) Exploitation
        qs = self.q_table[sid]
        if qs["A"] > qs["B"]:
            return "A"
        if qs["B"] > qs["A"]:
            return "B"
        return random.choice(["A", "B"])

    def calculate_reward(self, sid: str, action: Action) -> float:
        # Reward = dot(scenario_reward_vector, culture_weight_vector) * 10
        scn = next(s for s in self.scenarios if s.sid == sid)
        r_vec = scn.rewards[action]
        reward = sum(float(r_vec.get(k, 0.0)) * float(self.weights.get(k, 0.0)) for k in FRAMEWORKS) * 10.0
        return float(reward)

    def update(self, sid: str, action: Action, reward: float) -> None:
        old_val = float(self.q_table[sid][action])
        self.q_table[sid][action] = old_val + self.lr * (float(reward) - old_val)

    def decay_epsilon(self, decay: float = 0.99, min_epsilon: float = 0.01) -> None:
        self.epsilon = max(float(min_epsilon), float(self.epsilon) * float(decay))

    def get_avg_entropy(self) -> float:
        """
        Strategy entropy:
        Convert Q-values to probabilities via softmax and compute entropy.
        Lower entropy => more confident / stabilized policy
        Higher entropy => more uncertain / exploring
        """
        entropies: List[float] = []
        for sid, action_vals in self.q_table.items():
            qs = np.array([action_vals["A"], action_vals["B"]], dtype=float)

            # Numerically stable softmax
            qs = qs - np.max(qs)
            exp_qs = np.exp(qs)
            probs = exp_qs / np.sum(exp_qs)

            entropies.append(float(scipy_entropy(probs)))

        return float(np.mean(entropies)) if entropies else 0.0


# =========================
# 4. Analysis utilities
# =========================
def calculate_diversity(actions_list: List[Action]) -> float:
    """
    Diversity score in [0, 1]:
      - 1.0 when A/B are balanced (50/50)
      - 0.0 when fully biased (all A or all B)
    """
    if not actions_list:
        return 0.0
    a_count = actions_list.count("A")
    ratio = a_count / len(actions_list)
    return float(1.0 - (2.0 * abs(0.5 - ratio)))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_simulation(
    culture_name: str,
    weights: Dict[str, float],
    episodes: int,
    custom_scenarios: List[Scenario],
    seed: int,
    learning_rate: float = 0.1,
    epsilon: float = 0.5,
) -> pd.DataFrame:
    set_global_seed(seed)

    agent = SimpleEGreedyAgent(
        name=culture_name,
        weights=weights,
        scenarios=custom_scenarios,
        learning_rate=learning_rate,
        epsilon=epsilon,
    )

    history = {
        "episode": [],
        "reward": [],
        "diversity": [],
        "entropy": [],
    }

    progress = st.progress(0)
    for ep in range(int(episodes)):
        ep_actions: List[Action] = []
        ep_reward = 0.0

        for scn in custom_scenarios:
            action = agent.get_action(scn.sid)
            reward = agent.calculate_reward(scn.sid, action)
            agent.update(scn.sid, action, reward)

            ep_actions.append(action)
            ep_reward += reward

        agent.decay_epsilon()

        history["episode"].append(ep + 1)
        history["reward"].append(float(ep_reward))
        history["diversity"].append(calculate_diversity(ep_actions))
        history["entropy"].append(agent.get_avg_entropy())

        # Update progress ~ every 10 episodes (and last)
        if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
            progress.progress((ep + 1) / episodes)

    progress.empty()
    return pd.DataFrame(history)


# =========================
# 5. UI
# =========================
st.title("🎓 AI 윤리 의사결정 시뮬레이터 | 3팀")
st.markdown(
    """
이 앱은 문화권별 가치 가중치에 따라 AI 에이전트가 윤리적 딜레마에서
어떤 선택을 학습하는지 보여 주는 예시 시뮬레이터입니다.

1. **환경 설정**: 선택지별 보상값 조정
2. **에이전트 설정**: 문화권과 가치 가중치 선택
3. **결과 분석**: 보상, 선택 다양성, 전략 엔트로피 확인
"""
)

# --- Sidebar: Agent setup ---
st.sidebar.header("👤 2. 에이전트 설정")

FRAMEWORK_LABELS = {"emotion": "감정", "social": "사회", "moral": "도덕", "identity": "정체성"}
CULTURE_LABELS = {
    "USA": "미국", "CHINA": "중국", "EUROPE": "유럽", "KOREA": "한국",
    "LATIN_AM": "라틴아메리카", "MIDDLE_E": "중동", "AFRICA": "아프리카",
}

culture_presets = get_culture_presets()
selected_culture = st.sidebar.selectbox(
    "문화권 예시",
    list(culture_presets.keys()),
    index=list(culture_presets.keys()).index("MIDDLE_E"),
    format_func=lambda code: CULTURE_LABELS.get(code, code),
)
st.sidebar.caption("문화권 가중치는 제공된 예시값이며, 3팀의 조사 결과가 아닙니다.")

episodes = st.sidebar.slider("학습 횟수", 100, 1000, 300, step=50)
seed = st.sidebar.number_input("난수 시드 (결과 재현용)", min_value=0, max_value=1_000_000, value=42, step=1)

st.sidebar.subheader("가치 가중치 조정")
mod_weights: Dict[str, float] = {}
culture_defaults = culture_presets[selected_culture]
for k in FRAMEWORKS:
    mod_weights[k] = st.sidebar.slider(FRAMEWORK_LABELS[k], 0.0, 1.0, float(culture_defaults.get(k, 0.0)))

final_weights = normalize_weights(mod_weights)

st.sidebar.markdown("---")
st.sidebar.caption("정규화된 가중치 (합계 1.0):")
st.sidebar.json({FRAMEWORK_LABELS[k]: round(v, 3) for k, v in final_weights.items()})

# --- Main: Environment setup ---
st.header("🌍 1. 환경 설정 (시나리오 보상값)")
st.info(
    "네 가지 가치 기준(감정, 사회, 도덕, 정체성)에 따라 각 선택지의 "
    "보상값을 -1.0부터 1.0까지 조정하세요."
)

DEFAULT_SCENARIOS = get_default_scenarios()
custom_scenarios: List[Scenario] = []

tabs = st.tabs([s.title for s in DEFAULT_SCENARIOS])
for i, (tab, default_scn) in enumerate(zip(tabs, DEFAULT_SCENARIOS)):
    with tab:
        st.markdown(f"> **상황:** {default_scn.setup}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"### 🅰 {default_scn.options['A']}")
            r_a = dict(default_scn.rewards["A"])
            for fw in FRAMEWORKS:
                r_a[fw] = st.slider(
                    f"[A] {FRAMEWORK_LABELS[fw]}",
                    -1.0,
                    1.0,
                    float(r_a.get(fw, 0.0)),
                    0.1,
                    key=f"s{i}a_{fw}",
                )

        with col_b:
            st.markdown(f"### 🅱 {default_scn.options['B']}")
            r_b = dict(default_scn.rewards["B"])
            for fw in FRAMEWORKS:
                r_b[fw] = st.slider(
                    f"[B] {FRAMEWORK_LABELS[fw]}",
                    -1.0,
                    1.0,
                    float(r_b.get(fw, 0.0)),
                    0.1,
                    key=f"s{i}b_{fw}",
                )

        custom_scenarios.append(
            Scenario(
                sid=default_scn.sid,
                title=default_scn.title,
                setup=default_scn.setup,
                options=default_scn.options,
                rewards={"A": r_a, "B": r_b},
            )
        )

# --- Run analysis ---
st.divider()
st.header("🚀 3. 시뮬레이션과 결과 분석")

if st.button("시뮬레이션 실행", type="primary"):
    with st.spinner("AI 에이전트 학습 중..."):
        df = run_simulation(
            culture_name=selected_culture,
            weights=final_weights,
            episodes=int(episodes),
            custom_scenarios=custom_scenarios,
            seed=int(seed),
        )

    st.success("학습이 완료되었습니다!")

    display_df = df.rename(columns={
        "episode": "학습 회차", "reward": "총 보상",
        "diversity": "선택 다양성", "entropy": "전략 엔트로피",
    })

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📈 총 보상")
        st.caption("에이전트가 가치 가중치에 맞는 선택을 반복할수록 높아집니다.")
        st.line_chart(display_df, x="학습 회차", y="총 보상")

    with col2:
        st.markdown("#### 📉 전략 엔트로피")
        st.caption("값이 낮을수록 선택 전략이 더 안정되었음을 뜻합니다.")
        st.line_chart(display_df, x="학습 회차", y="전략 엔트로피")

    with col3:
        st.markdown("#### 🔀 선택 다양성")
        st.caption("1에 가까울수록 A와 B를 비슷한 비율로 선택했습니다.")
        st.line_chart(display_df, x="학습 회차", y="선택 다양성")

    st.markdown("---")
    st.subheader("🔗 상관관계 분석: 선택 다양성과 보상")

    r_val, p_val = pearsonr(df["diversity"], df["reward"])

    c_plot, c_stat = st.columns([2, 1])
    with c_plot:
        points = alt.Chart(display_df).mark_circle(opacity=0.65).encode(
            x=alt.X("선택 다양성:Q", title="선택 다양성 (0=한쪽 편중, 1=균형)"),
            y=alt.Y("총 보상:Q", title="총 보상"),
            tooltip=["학습 회차", "선택 다양성", "총 보상"],
        )
        chart = points
        if display_df["선택 다양성"].nunique() > 1:
            trend = alt.Chart(display_df).transform_regression(
                "선택 다양성", "총 보상"
            ).mark_line(color="red", strokeDash=[6, 4]).encode(
                x="선택 다양성:Q", y="총 보상:Q"
            )
            chart = points + trend
        st.altair_chart(
            chart.properties(title=f"선택 다양성과 총 보상 (r={r_val:.2f})"),
            width="stretch",
        )

    with c_stat:
        st.metric("피어슨 상관계수 r", f"{r_val:.3f}")
        st.metric("유의확률 p값", f"{p_val:.3e}")
        if r_val > 0.3:
            st.success("✅ 양의 상관관계: 선택 다양성이 높을수록 보상이 증가하는 경향이 있습니다.")
        elif r_val < -0.3:
            st.warning("⚠️ 음의 상관관계: 선택 다양성이 낮을수록 보상이 증가하는 경향이 있습니다.")
        else:
            st.info("⏺ 뚜렷한 선형 상관관계가 확인되지 않았습니다.")

    with st.expander("📥 학습 기록 내려받기"):
        st.dataframe(display_df.head())
        st.download_button("CSV 내려받기", df.to_csv(index=False), "ai_ethics_data.csv")