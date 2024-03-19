# POLICY EVALUATION

## AIM
To evaluate and compare the performance of two policies using policy evaluation.

## PROBLEM STATEMENT

1. Given a set of states, actions, and transition probabilities, we are given two policies.
2. We want to evaluate the performance of the two policies by computing their state-value functions.
3. The policy with the higher state-value function is considered to be the better policy.


## POLICY EVALUATION FUNCTION
```
#Name : Mohamed Asil
#Reg no: 212222230080

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V=np.zeros(len(P))
      for s in range(len(P)):
        for prob,next_state,reward,done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V)):
        break
    return V
```

## OUTPUT:

![image](https://github.com/Asilsathik/rl-policy-evaluation/assets/119476247/74c9c7ed-1458-4937-8a0e-ef1127319928)


## RESULT:
![image](https://github.com/Asilsathik/rl-policy-evaluation/assets/119476247/a4b5072d-2667-4ff9-bbd7-091c11359784)

