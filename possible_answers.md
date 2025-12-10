# Possible Answer Interpretations for Time Series Questions

## Question 10: MA{5} Weight

**If MA{5} uses 5 points (standard):** 1/5

**If MA{5} uses 6 points (j=0 to w=5):** 1/6

## Question 11: MA{5} Approximations

### Centered MA{5} with 5 points (t-2, t-1, t, t+1, t+2):
- t=7: 105.6941
- t=23: 114.2475
- t=79: 105.7972

### Trailing MA{5} with 5 points (t-4, t-3, t-2, t-1, t):
- t=7: 107.8098
- t=23: 111.1467
- t=79: 107.8447

### Forward MA{5} with 5 points (t, t+1, t+2, t+3, t+4):
- t=7: 106.6974
- t=23: 115.7049
- t=79: 107.3509

### Centered MA{5} with 6 points (t-2, t-1, t, t+1, t+2, t+3):
- t=7: 106.1063
- t=23: 113.1938
- t=79: 106.7097

## Question 12: EMA{3, 0.32} Weights

### Normalized (sum to 1):
- w0 = 0.4070
- w1 = 0.2768
- w2 = 0.1882
- w3 = 0.1280

### Unnormalized (standard exponential):
- w0 = 0.3200
- w1 = 0.2176
- w2 = 0.1480
- w3 = 0.1006

## Question 13: EMA{3, 0.32} Approximation at t=35

### With normalized weights:
114.1011

### With unnormalized weights:
89.7047

---

## Recommendations:
1. Check your course materials for the EXACT definition of MA{w} and EMA{w,a}
2. Look for whether "centered" vs "trailing" is specified
3. Check if weights should be normalized
4. Verify the quiz isn't expecting a different number of decimal places
5. Consider asking your professor for clarification on the methodology
