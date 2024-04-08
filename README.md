Development Status :: 3 - Alpha

# LLM Translator on Hugging-face models <img alt="PyPI" src="https://img.shields.io/pypi/v/transllm?color=black">

<p align="right">
    <a><img alt="PyPI package" src="https://img.shields.io/badge/pypi-transllm-black"></a>
    <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fdsdanielpark%2Fhf-transllm&count_bg=%23000000&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</p>


LLMtranslator translates and generates text in multiple languages using LLMs(Large Language Models) on hugging-face models.

![](assets/transllm.png)

### Introducing hf-transllm: Unlock the Power of Multilingual Exploration

Discover the hf-transllm package, a seamless integration of Hugging Face's inference module and translation APIs. Overcome limitations in retraining and evaluating large language models in different languages. Explore diverse results effortlessly, leveraging translation API services. Emphasizing versatility over efficiency, hf-transllm enables you to delve into the outcomes of Hugging Face's models in various languages.


<br>


<br>

## Installation
```
pip install transllm
```
```
pip install git+https://github.com/dsdanielpark/hf-trnasllm.git
```

<br>

## CLI
If you wish to use CLI:
```
git clone https://github.com/dsdanielpark/hf-transllm
cd hf-transllm
pip install -r requirements.txt
```
```bash
python main.py --hfmodel <openlm-research/open_llama_3b> --lang <ko> --translator <google>
```
There can be issues with various dependencies such as Hugging Face's Transformers, SentencePiece, Torch, and CUDA. Please set up the appropriate environment by searching online.
<br>

<br>

## Usage    

*Simple Usage*
```python
from transllm import LLMtranslator

open_llama3b_kor = LLMtranslator('openlm-research/open_llama_3b', target_lang='ko', translator='google') # Korean

trnaslated_answer = open_llama3b_kor.generate("나와 내 동년배들이 좋아하는 뉴진스에 대해서 알려줘")
print(trnaslated_answer)
```

<br>

## Translation API Integration Guide [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1117ikGEmU4FncBDl1xCC2IhPPDOr75lX?usp=sharing)
### Google Translator
- Official Google Translation API Supported Languages: [Google Cloud Languages](https://cloud.google.com/translate/docs)
- Unofficial Google Translator (Non-profit/Testing) Supported Languages: [Deep Translator Constants](https://github.com/nidhaloff/deep-translator/blob/master/deep_translator/constants.py)

*Usage Example*
```python
from transllm import LLMtranslator

model_path = 'openlm-research/open_llama_3b'
# Alternative Models: 'openlm-research/open_llama_7b', 'openlm-research/open_llama_13b'

# For official Google Cloud Translation
# open_llama3b_kor = LLMtranslator(model_path, target_lang='ko', translator='google_official', google_api_key='YOUR_API_KEY')

# For unofficial testing
open_llama3b_kor = LLMtranslator(model_path, target_lang='ko', translator='google')

prompt = "Translate this text"
translated_answer = open_llama3b_kor.generate(prompt)
print(translated_answer)
```

<br>

### DeepL
- Supported Languages of DeepL: [DeepL Languages](https://www.deepl.com/pro/select-country?cta=header-pro-button/#developer)

*Pre-prompt Translation Example*
```python
from transllm import LLMtranslator

model_path = 'openlm-research/open_llama_3b'

# Choose your Translation Service API
open_llama3b_kor = LLMtranslator(model_path, target_lang='ES', translator='deepl', deepl_api='YOUR_DEEPL_API')
# Alternative setups for OpenAI and Bard

prompt = "Translate this text"
translated_response = open_llama3b_kor.generate(prompt)
print(translated_response)
```
<br>

### OpenAI, Anthropic, Gemini Translation Services
In progress


<br>

## Customized Inference
Customizing the inference process for unique tokenizers or inference needs is possible. For advanced customization, add a translation module before and after the Hugging Face inference code.

### Custom Inference Example
```python
import torch
from trnasllm import LLMtranslator

class MyLLMtranslator(LLMtranslator):
    def __init__(self, model_path, target_lang="ko", translator="google", **kwargs):
        super().__init__(model_path=model_path, target_lang=target_lang, translator=translator, **kwargs)

    def inference(self, prompt: str) -> str:
        # Custom logic here
        return custom_logic(prompt)
```

### Open LLM Examples on Hugging Face
- Several models listed for reference (e.g., `hf-internal-testing/tiny-random-gpt2`, `EleutherAI/gpt-neo-125m`).

### Google Translator Note
- Official use of Google Translate is chargeable. Use `translator="google_official"` and provide `google_api_key`. Unofficial testing should use `translator="google"`. Refer to the [official documentation](https://cloud.google.com/translate) for more details.


<br><br>

## [FAQs](./documents/FAQs.md)
You can find most help on the [FAQ](https://github.com/dsdanielpark/hf-transllm/blob/main/documents/README_FAQ.md) and [Issue](https://github.com/dsdanielpark/hf-transllm/issues) pages. 

## Contributors

For detailed guidance on contributions, please refer to the [contribution guide](https://github.com/dsdanielpark/open-interview/blob/main/docs/contributions.md). We appreciate your interest in contributing and look forward to your valuable input. 

Thank you for supporting our project.



## License ©️ 
[MIT](https://opensource.org/license/mit/) license, 2024. We hereby strongly disclaim any explicit or implicit legal liability related to our works. Users are required to use this package responsibly and at their own risk. This project is a personal initiative and is not affiliated with or endorsed by Google, DeepL, Oepn AI and Anthropic.


## Contacts
- Core maintainer: [Daniel Park, South Korea](https://github.com/DSDanielPark) <br>
- E-mail: parkminwoo1991@gmail.com <br>

## Reference 
[1] https://huggingface.co/docs/api-inference/index <br>
  
<br>
            

  
*Copyright (c) 2023 MinWoo Park, South Korea*<br>
