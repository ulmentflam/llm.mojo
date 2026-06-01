# LLM.mojo

LLMs in Mojo without the need for PyTorch or CPython. Inspired by karpathy's [llm.c](https://github.com/karpathy/llm.c), with a focus on proving out the viability of autograd in pure Mojo syntax. The focus is to reproduce GPT-2 and GPT-3 alongside a parallel PyTorch reference in `train_gpt*.py`.

A personal goal of this project is to write all code without using any LSPs or LLMs, writing every algorithm (forward and backpropagation) from scratch. I received feedback recently that my "coding and math expertise" are not strong enough, and building out this framework is how I intend to strengthen those skills. Just like writing a compiler, writing the fundamentals of generative models from scratch sharpens both engineering and mathematics.

As part of that goal, I will be leveraging Nvidia Nsight and Perfetto for performance analysis and comparison against my PyTorch implementation of GPT-2. As the project evolves, I will include benchmarking results and other insights into the performance comparisons between Mojo, PyTorch, and even karpathy's C implementation.


# Thanks

A special thanks to https://github.com/dorjeduck/llm.mojo and @dorjeduck for writing the original implementation of llm.mojo in Mojo 25.5.
