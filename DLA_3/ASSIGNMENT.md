# Official Assignment - DLA 3

> Official assignment provided by the course instructor.  
> This file is included to document the requirements addressed by the submitted work.

In questa attività "flipped", familiarizzerete con le basi del Deep Reinforcement Learning, concentrandovi sui metodi Policy Gradient. In questa sezione sono allegati:

Un breve notebook Jupyter (e il relativo video) con un'implementazione rapida e semplice dell'algoritmo Policy Gradient più semplice (REINFORCE) per addestrare una rete neurale stocastica basata sull'esperienza.

Il notebook di laboratorio vero e proprio con gli esercizi.

Per eseguire i notebook è necessario un ambiente con alcune dipendenze aggiuntive. Ho creato il mio ambiente (su Linux) con:

conda create -n DRL -c conda-forge torch gymnasium pytorch-gpu matplotlib pygame jupyterlab jupyter
oppure, se si utilizza uv:

uv init
uv add gymnasium torch matplotlib pygame jupyterlab jupyter
Nota: l'accelerazione GPU in PyTorch non è necessaria per eseguire il notebook di esempio, quindi tutto funzionerà correttamente anche con la versione non GPU di PyTorch.

Nota inoltre: se l'esperienza mi ha insegnato qualcosa, se si utilizza Windows si avranno grossi problemi a far funzionare gli ambienti Farama Gymnasium. Questo perché i due ambienti che utilizzeremo richiedono il simulatore fisico MuJoCo e la libreria di rilevamento delle collisioni Box2D. Pertanto, se si utilizza Windows, suggerisco di seguire le seguenti istruzioni:

Non utilizzare Windows.
Siate contenti di non utilizzare Windows.

Più seriamente, questo video contiene consigli più pertinenti su come far funzionare Gymnasium su Windows. Ma onestamente, è meglio creare e utilizzare una partizione Linux (o una macchina virtuale) invece di usare direttamente Windows. Questo notebook non funzionerà su Google Colab perché necessita di creare finestre locali per la visualizzazione.

Nota finale: se c'è un laboratorio in cui non dovete assolutamente usare un Jupyter Notebook per le vostre implementazioni, è proprio questo. A causa della visualizzazione degli agenti in esecuzione, il notebook è piuttosto fragile e semplicemente non funzionerà su Google Colab.

# Deep Reinforcement Learning Laboratory

In this laboratory session we will work on getting more advanced versions of Deep Reinforcement Learning algorithms up and running. Deep Reinforcement Learning is **hard**, and getting agents to stably train can be frustrating and requires quite a bit of subtlety in analysis of intermediate results. We will start by refactoring (a bit) my implementation of `REINFORCE` on the [Cartpole environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/). 

## Exercise 1: Improving my `REINFORCE` Implementation (warm up)

In this exercise we will refactor a bit and improve some aspects of my `REINFORCE` implementation. 

**First Things First**: Spend some time playing with the environment to make sure you understand how it works.

**Next Things Next**: Now get your `REINFORCE` implementation working on the environment. You can import my (probably buggy and definitely inefficient) implementation here. Or even better, refactor an implementation into a separate package from which you can `import` the stuff you need here. 

**Last Things Last**: My implementation does a **super crappy** job of evaluating the agent performance during training. The running average is not a very good metric. Modify my implementation so that every $N$ iterations (make $N$ an argument to the training function) the agent is run for $M$ episodes in the environment. Collect and return: (1) The average **total** reward received over the $M$ iterations; and (2) the average episode length. Analyze the performance of your agents with these new metrics.

----
## Exercise 2: `REINFORCE` with a Value Baseline (warm up)

In this exercise we will augment my implementation (or your own) of `REINFORCE` to subtract a baseline from the target in the update equation in order to stabilize (and hopefully speed-up) convergence. For now we will stick to the Cartpole environment.

**First Things First**: Recall from the slides on Deep Reinforcement Learning that we can **subtract** any function that doesn't depend on the current action from the q-value without changing the (maximum of our) objecttive function $J$:  

$$ \nabla J(\boldsymbol{\theta}) \propto \sum_{s} \mu(s) \sum_a \left( q_{\pi}(s, a) - b(s) \right) \nabla \pi(a \mid s, \boldsymbol{\theta}) $$

In `REINFORCE` this means we can subtract from our target $G_t$:

$$ \boldsymbol{\theta}_{t+1} \triangleq \boldsymbol{\theta}_t + \alpha (G_t - b(S_t)) \frac{\nabla \pi(A_t \mid s, \boldsymbol{\theta})}{\pi(A_t \mid s, \boldsymbol{\theta})} $$

Since we are only interested in the **maximum** of our objective, we can also **rescale** our target by any function that also doesn't depend on the action. A **simple baseline** which is even independent of the state -- that is, it is **constant** for each episode -- is to just **standardize rewards within the episode**. So, we **subtract** the average return and **divide** by the variance of returns:

$$ \boldsymbol{\theta}_{t+1} \triangleq \boldsymbol{\theta}_t + \alpha \left(\frac{G_t - \bar{G}}{\sigma_G}\right) \nabla  \pi(A_t \mid s, \boldsymbol{\theta}) $$

This baseline is **already** implemented in my implementation of `REINFORCE`. Experiment with and without this standardization baseline and compare the performance. We are going to do something more interesting.

**The Real Exercise**: Standard practice is to use the state-value function $v(s)$ as a baseline. This is intuitively appealing -- we are more interested in updating out policy for returns that estimate the current **value** worse. Our new update becomes:

$$ \boldsymbol{\theta}_{t+1} \triangleq \boldsymbol{\theta}_t + \alpha (G_t - \tilde{v}(S_t \mid \mathbf{w})) \frac{\nabla \pi(A_t \mid s, \boldsymbol{\theta})}{\pi(A_t \mid s, \boldsymbol{\theta})} $$

where $\tilde{v}(s \mid \mathbf{w})$ is a **deep neural network** with parameters $w$ that estimates $v_\pi(s)$. What neural network? Typically, we use the **same** network architecture as that of the Policy.

**Your Task**: Modify your implementation to fit a second, baseline network to estimate the value function and use it as **baseline**. 

-----

## Exercise 3: Going Deeper

As usual, pick **AT LEAST ONE** of the following exercises to complete.

### Exercise 3.1: Solving Cartpole and Lunar Lander with A2C (easy)

Implement the Advantage Actor-Critic (A2C) algorithm and use it to solve both `Cartpole` (to validate your implementation) and the [Lunar Lander Environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/). This environment is a little bit harder than Cartpole, but not much. Make sure you perform the appropriate types of analyses to quantify and qualify the performance of your agents.

**Why choose this exercise?** A2C is a good, "pure" TD-based reinforcement learning algorithm and is the basis for more advanced Policy Gradient approaches. Having a good understanding of it is a good starting point for diving into the state-of-the-art in on-policy Deep Reinforcement Learning.
