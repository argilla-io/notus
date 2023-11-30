# 💨 Notus Handbook

Notus is a collection of fine-tuned models using SFT, DPO, SFT+DPO, and/or any other RLHF techniques; but always aiming to follow a data-first approach, since that's what we do best at Argilla.

Notus models are intended to be used as assistants via chat-like applications, and are evaluated with Chat (MT-Bench, AlpacaEval) and Academic (Open LLM Leaderboard) benchmarks for a direct comparison with other similar LLMs.

Being able to fine-tune LLMs while still keeping a data-first approach wouldn't have been possible without the inestimable help of the open source community and all the amazing resources out there intended for the general public. We are very grateful for that, and we hope that our work can be useful for others as well.

🎩 h/t HuggingFace H4 team for their amazing work with [`alignment-handbook`](https://github.com/huggingface/alignment-handbook), and also for the fruitful dicussions we had with them and their support.

## News

* 30th November, 2023: **Notus 7B v1 is released! 🎉** Using the same DPO fine-tuning approach as Zephyr 7B Beta, but changing the data source from UltraFeedback to binarize it using the average of the different criterias, instead of the critique score. Notus 7B improved in both AlpacaEval and LM Eval Harness compared to Zephyr 7B Beta, while for MT-Bench the results were a tiny bit behind. More information at [`notus-handbook/v1`](./v1/).

## Resources

### 🤗 HuggingFace Hub Collection

<div align="center">
  <img width="702" alt="image" src="https://github.com/argilla-io/notus-7b-dpo/assets/36760800/49bddbd2-ecfc-46d6-8d1d-1cb760dfe08b">
  <p>Available at: https://huggingface.co/collections/argilla/notus-7b-v1-655529d7c73cb6c830e9555a</p>
</div>

### 💬 Chat UI

<div align="center">
  <img width="1624" alt="image" src="https://github.com/argilla-io/notus-7b-dpo/assets/36760800/a950f7f2-74ea-4873-a314-3afd1d4d7ac8">
  <p>Chat with Notus at https://argilla-notus-chat-ui.hf.space/ (powered by https://github.com/huggingface/chat-ui)</p>
</div>

## Citation

```bibtex
@misc{alignment_handbook2023,
  author = {Alvaro Bartolome and Gabriel Martin and Daniel Vila},
  title = {Notus Handbook},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/argilla-io/notus-handbook}}
}
```
> [!NOTE]
> Alphabetically ordered by last name due to equal contribution.