# TGI
The local tokenizer config can be supplied to TGI through the flag `--tokenizer-config-path`, documented [here](https://huggingface.co/docs/text-generation-inference/basic_tutorials/launcher#tokenizerconfigpath).

# vLLM
A local chat template can be supplied to vLLM through the flag `--chat-template`. It is not explicitly documented, but can be found mentioned in GitHub Issues relating to the topic.

# Llama-2 models on TGI
There is a [known bug with TGI](https://github.com/huggingface/text-generation-inference/issues/1534) in which the default `tokenizer_config.json` is not handled properly by TGI by applying chat templating. While this is resolved, we are using a modified `tokenizer_config.json` that is compatible with TGI. Note that the chat templating jinja itself the same, with the exception of removing 2 calls to `.strip()`, which TGI reports errors on.

For reference, here is the original unmodified chat template:
```
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = false %}
{% endif %}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 and system_message != false %}
        {% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + content.strip() + ' ' + eos_token }}
    {% endif %}
{% endfor %}
```

We also note that the `eos_token` and `bos_token` are originally provided as maps, but the TGI implementation only accepts a string. So we also modify them to only contain the `content` string.

For reference, here is the original unmodified `tokenizer_config.json`:
```
{
    "add_bos_token": true,
    "add_eos_token": false,
    "bos_token": {
      "__type": "AddedToken",
      "content": "<s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false
    },
    "chat_template": "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content + ' ' + eos_token }}{% endif %}{% endfor %}",
    "clean_up_tokenization_spaces": false,
    "eos_token": {
      "__type": "AddedToken",
      "content": "</s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false
    },
    "legacy": false,
    "model_max_length": 1000000000000000019884624838656,
    "pad_token": null,
    "padding_side": "right",
    "sp_model_kwargs": {},
    "tokenizer_class": "LlamaTokenizer",
    "unk_token": {
      "__type": "AddedToken",
      "content": "<unk>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false
    }
}
```

# Mistral with chat templating
Mistral for chatting has not been explicitly trained using a distinct system prompt. Therefore, the default Mistral `tokenizer_config.json` explicitly assumes that the system role does not exist. To keep our benchmarks consistent across models, we reenginered the original Mistral chat template to account for a system prompt. We simply preppend the system prompt to the first user prompt in a given conversation.
