# 수식 읽기용 plain-Markdown 요약

일부 Markdown viewer는 MathJax/KaTeX를 지원하지 않습니다. 그런 viewer에서는
`\[ ... \]`, `\sigma`, `\dot` 같은 LaTeX 명령이 수식으로 변환되지 않고 그대로
문자로 표시됩니다. 아래는 별도 수식 플러그인 없이 읽을 수 있도록 같은 내용을
ASCII/Unicode 표기로 다시 쓴 요약입니다.

## 공통 기호

```text
P_act,t       actuator pressure at time t
P_ch,t        chamber pressure at time t
r_t           pressure reference
u_t           valve action
dt            simulator sample time = 1 / f_s
H_t           finite observation/action history
```

## HRR

```text
pressure_rate_t = (P_t - P_(t-1)) * f_s

reserve_pos_t = P_ch_pos,t - P_act_pos,t
reserve_neg_t = P_act_neg,t - P_ch_neg,t

HRR = [four pressure rates, reserve_pos, reserve_neg]
```

HRR은 새 센서 정보가 아니라 history에서 계산 가능한 좌표다. lib3에는 이미
`history_rate_reserve` 실험이 있었고, 현재 코드의 `hrr`는 이 feature의 alias다.

## e_dot

```text
e_t     = r_t - P_act,t
e_dot_t = (e_t - e_(t-1)) * f_s
        = r_dot_t - P_act_dot,t
```

`e_dot`은 error가 줄어드는지 커지는지를 알려주는 2개 tracking derivative다.
reserve나 integral은 포함하지 않는다.

## PIDE

```text
P term = e_t
I term = sum_i e_i * dt       # finite history integral
D term = e_dot_t

PIDE = [P term, I term, D term] for positive/negative actuator
      = 6 features in lib3
```

PIDE는 tracking error 중심의 task-specific summary다. 현재 구현은 raw history를
삭제하지 않고 P/I/D feature를 추가하므로, 정확한 recurrent PIDE encoder가 아니라
PIDE-style fixed feature baseline이다.

## GPIDE

현재 프로젝트의 고정 GPIDE-style feature는 다음이다.

```text
delta_obs_t = obs_t - obs_(t-1)      # lib3: four pressure differences
delta_u_t   = u_t - u_(t-1)

GPIDE-lite = [delta_obs_t, u_t, delta_u_t]
           = 4 + 6 + 6 = 16 features in lib3
```

따라서 lib3에서 `delta_obs_t`는 pressure difference이고, `p_dot`와 단지 sample
frequency/scaling이 다른 같은 방향의 좌표다. action history까지 이미 넣고
있다면 `u_t`와 `delta_u_t`도 일부 중복된다. 그러므로 현재 구현의 GPIDE는
`history + p_dot`와 크게 다르지 않으며 primary novelty로 주장하면 안 된다.

## 관계 한 줄 요약

```text
HRR  = plant pressure trend + pneumatic reserve
e_dot = tracking error trend
PIDE = tracking error + accumulated error + error trend
GPIDE-lite = generic observation difference + action change
```

## history에서 rate를 넣어도 표현력이 증가하지 않는 이유

```text
D_t = L * H_t

first_layer(H_t, D_t)
  = W_H * H_t + W_D * D_t + b
  = (W_H + W_D * L) * H_t + b
```

따라서 explicit rate는 exact information을 추가하지 않는다. finite MLP가
차분/스케일링을 직접 학습할 필요를 줄이는 coordinate/conditioning bias가
핵심이다.

## Lambda discrepancy

```text
G_lambda[t] = reward[t] + gamma * (
                  (1 - lambda) * V[t+1]
                  + lambda * G_lambda[t+1]
              )

discrepancy = mean_abs(G_lambda_low - G_lambda_high)
```

이는 representation의 value prediction이 얼마나 Markov-like한지 보는 보조
진단이지, closed-loop controller 성능 자체가 아니다.

