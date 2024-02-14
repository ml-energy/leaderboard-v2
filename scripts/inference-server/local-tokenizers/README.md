# TGI
The local tokenizer config can be supplied to TGI through the flag `--tokenizer-config-path`, documented [here](https://huggingface.co/docs/text-generation-inference/basic_tutorials/launcher#tokenizerconfigpath).

# vLLM
A local chat template can be supplied to vLLM through the flag `--chat-template`. It is not explicitly documented, but can be found mentioned in GitHub Issues relating to the topic.

# Llama-2 models on TGI
There is a [known bug with TGI](https://github.com/huggingface/text-generation-inference/issues/1534) in which the default `tokenizer_config.json` is not handled properly by TGI by applying chat templating. While this is resolved, we are using a modified `tokenizer_config.json` that is compatible with TGI. Note that the chat templating jinja itself the same, with the exception of removing any calls to `.split()`, which TGI reports errors on.

# Mistral with chat templating
Mistral for chatting has not been explicitly trained using a distinct system prompt. Therefore, the default Mistral `tokenizer_config.json` explicitly assumes that the system role does not exist. To keep our benchmarks consistent across models, we reenginered the original Mistral chat template to account for a system prompt. We simply preppend the system prompt to the first user prompt in a given conversation.
