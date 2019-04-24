# DeepTraffic python environment

My (not complete) DeepTraffic training environment, includes the following:
* Python gym-compatible environment of DeepTraffic
* Unit tests :)
* Training and playing code (simple DQN model using my ptan lib: https://github.com/Shmuma/ptan/)
* PyTorch to ConvNetJS converter. This converter is not finished, contribution is very welcome.

In terms of DeepTraffic environment dynamics, I've tried to reproduce it as close as possible
(including reverse engineering obfuscated JS code (c:), but, of course it might differ.
Final criteria, of course should be ability of trained policy to play successfully in DeepTraffic competition console,
but I'm still not there :).
