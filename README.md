# Learning-to-Optimize 📈
A (mostly faithful) implementation of the 2016 paper [Learning to Optimize](https://arxiv.org/abs/1606.01885). You can find a more casual explanation in [this blog post](https://bair.berkeley.edu/blog/2017/09/12/learning-to-optimize-with-rl/).

The main idea behind the paper is simple: instead of using handcrafted optimization algorithms, why not learn them? The paper presents a framework for learning optimization algorithms for machine learning objectives. They cast this as a reinforcement learning problem, where the algorithm is the policy, and the reward is the negative objective function. Check out the blog post or paper for more details.

Disclaimer: This was a final project for a class. You can find the hastily written report [here](FoRL_Final_Project_Report.pdf), which contains details about implementation and a few performance plots. Unfortunately, there wasn't time to implement the Guided Policy Search algorithm used in the original work (more [here](http://proceedings.mlr.press/v28/levine13.html) and [here](https://papers.nips.cc/paper/2014/hash/6766aa2750c19aad2fa1b32f36ed4aee-Abstract.html)), so we tested with standard RL algorithms like A2C and PPO instead. We obtained similar performance as the original paper on most objectives, but performed worse when optimizing the 2-layer MLP. However we also performed little hyperparameter tuning, and think performance could be significantly improved.

I hope this can be a useful reference for others looking to explore optimizer learning. And of course, feel free to reach out to [me](mailto:slocumstewy@gmail.com)!

## Getting Started 🤩
To install requirements, run:
```bash
$ pip install -r requirements.txt
```

To run the code, check out the notebook `experiments.ipynb`.

Format with `make format`, lint with `make lint`.
