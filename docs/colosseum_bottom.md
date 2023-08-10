### Terms of use

By using our service, you agree to these Terms of Use and accept that the Service provides an approximate estimation of model inference energy usage for research purposes only. We are not liable for any damages or loss incurred by you or any third party arising from the use of the Service. It may generate offensive content and offers limited safety measures, thus should not be used for any illegal, harmful, violent, racist, or sexual purposes. The service collects user dialogue data and voting results. We reserve the right to distribute the dataset in the future.

### Technical details

- We allow models to generate only up to 512 new tokens. Due to this, some responses may be cut off in the middle.
- Tokens are sampled from the model output with `temperature` 1.0, `repetition_penalty` 1.0, `top_k` 50, and `top_p` 0.95.
- Large models (>= 30B) run on two NVIDIA A40 GPUs with tensor parallelism, whereas other models run on one NVIDIA A40 GPU. We directly measure the energy consumption of these GPUs.

### Contact

Please direct general questions and issues related to the Colosseum to our GitHub repository's [discussion board](https://github.com/ml-energy/leaderboard/discussions).
You can find the ML.ENERGY initiative members in [our homepage](https://ml.energy#members).
